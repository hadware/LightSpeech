import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import configargparse
import hydra
import numpy as np
import torch
import tqdm
from tensorboardX import SummaryWriter

from lightspeech.core.optimizer import get_std_opt
from lightspeech.dataset import dataloader as loader
from lightspeech.dataset import valid_symbols
from lightspeech.optispeech import OptiSpeechGenerator
from lightspeech.utils.hparams import HParam
from lightspeech.utils.plot import generate_audio, plot_spectrogram_to_numpy
from lightspeech.utils.util import read_wav_np

BATCH_COUNT_CHOICES = ["auto", "seq", "bin", "frame"]
BATCH_SORT_KEY_CHOICES = ["input", "output", "shuffle"]

def load_model(config_path: str, config_name: str) -> OptiSpeechGenerator:
    # Charger la configuration
    with hydra.initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name)
        # Instancier le modèle
        model = instantiate(cfg)
        return model


def train(args, hp, hp_str, logger, vocoder):
    chckpt_dir : Path = hp.train.chkpt_dir
    outdir: Path = hp.train.outdir
    assets_dir: Path = outdir / 'assets'
    data_dir: Path = hp.data.data_dir
    ckckpt_path : Path = args.checkpoint_path
    log_dir : Path = hp.train.log_dir

    # makedir all previous folders
    chckpt_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if hp.train.ngpu > 0 else "cpu")

    dataloader = loader.get_tts_dataset(data_dir, hp.train.batch_size, hp)
    validloader = loader.get_tts_dataset(data_dir, 5, hp, True)

    idim = len(valid_symbols)
    odim = hp.audio.num_mels
    model_partial = load_model(hp.model_config_folder, hp.config_name)
    model = model_partial(odim)
    # set torch device
    model = model.to(device)
    print("Model is loaded ...")
    print("New Training")
    global_step = 0
    optimizer = get_std_opt(model, hp.model.adim, hp.model.transformer_warmup_steps, hp.model.transformer_lr)

    print("Batch Size :", hp.train.batch_size)

    num_params(model)

    log_path = log_dir / args.name
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_path))
    model.train()
    forward_count = 0
    # print(model)
    for epoch in range(hp.train.epochs):
        start = time.time()
        running_loss = 0
        j = 0

        pbar = tqdm.tqdm(dataloader, desc='Starting training...')
        for data in pbar:
            global_step += 1
            x, input_length, y, _, out_length, _, dur, e, p = data
            # x : [batch , num_char], input_length : [batch], y : [batch, T_in, num_mel]
            #             # stop_token : [batch, T_in], out_length : [batch]

            # NOTE: y = mel spectro
            loss, report_dict = model(x.cuda(),
                                      input_length.cuda(),
                                      y.cuda(),
                                      out_length.cuda(), # output audio length
                                      dur.cuda(), # phonemes durations
                                      e.cuda(), # energy
                                      p.cuda()) # pitch
            loss = loss.mean() / hp.train.accum_grad
            running_loss += loss.item()

            loss.backward()

            # update parameters
            forward_count += 1
            j = j + 1
            if forward_count != hp.train.accum_grad:
                continue
            forward_count = 0
            step = global_step

            # compute the gradient norm to check if it is normal or not
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.train.grad_clip)
            logging.debug('grad norm={}'.format(grad_norm))
            if math.isnan(grad_norm):
                logging.warning('grad norm is nan. Do not update model.')
            else:
                optimizer.step()
            optimizer.zero_grad()

            if step % hp.train.summary_interval == 0:
                pbar.set_description(
                    "Average Loss %.04f Loss %.04f | step %d" % (running_loss / j, loss.item(), step))

                for r in report_dict:
                    for k, v in r.items():
                        if k is not None and v is not None:
                            if 'cupy' in str(type(v)):
                                v = v.get()
                            if 'cupy' in str(type(k)):
                                k = k.get()
                            writer.add_scalar("main/{}".format(k), v, step)

            if step % hp.train.validation_step == 0:

                for valid in validloader:
                    x_, input_length_, y_, _, out_length_, ids_, dur_, e_, p_ = valid
                    model.eval()
                    with torch.no_grad():
                        loss_, report_dict_ = model(x_.cuda(), input_length_.cuda(), y_.cuda(), out_length_.cuda(),
                                                    dur_.cuda(), e_.cuda(),
                                                    p_.cuda())

                        mels_ = model.inference(x_[-1].cuda())  # [T, num_mel]

                    model.train()
                    for r in report_dict_:
                        for k, v in r.items():
                            if k is not None and v is not None:
                                if 'cupy' in str(type(v)):
                                    v = v.get()
                                if 'cupy' in str(type(k)):
                                    k = k.get()
                                writer.add_scalar("validation/{}".format(k), v, step)
                    break

                mels_ = mels_.T  # Out: [num_mels, T]
                writer.add_image('melspectrogram_target',
                                 plot_spectrogram_to_numpy(y_[-1].T.data.cpu().numpy()[:, :out_length_[-1]]),
                                 step, dataformats='HWC')
                writer.add_image('melspectrogram_prediction',
                                 plot_spectrogram_to_numpy(mels_.data.cpu().numpy()),
                                 step, dataformats='HWC')

                # print(mels.unsqueeze(0).shape)

                audio = generate_audio(mels_.unsqueeze(0),
                                       vocoder)  # selecting the last data point to match mel generated above
                audio = audio.cpu().float().numpy()
                audio = audio / (audio.max() - audio.min())  # get values between -1 and 1

                writer.add_audio(tag=f"generated {ids_[-1]}.wav",
                                 snd_tensor=torch.Tensor(audio),
                                 global_step=step,
                                 sample_rate=hp.audio.sample_rate)

                _, target = read_wav_np(hp.data.wav_dir + f"{ids_[-1]}.wav", sample_rate=hp.audio.sample_rate)

                writer.add_audio(tag=f" target {ids_[-1]}.wav ",
                                 snd_tensor=torch.Tensor(target),
                                 global_step=step,
                                 sample_rate=hp.audio.sample_rate)

                ##
            if step % hp.train.save_interval == 0:
                save_path = os.path.join(hp.train.chkpt_dir, args.name,
                                         '{}_fastspeech_{}_{}k_steps.pyt'.format(args.name, githash, step // 1000))

                torch.save({
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'step': step,
                    'hp_str': hp_str,
                    'githash': githash,
                }, save_path)
                logger.info("Saved checkpoint to: %s" % save_path)
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def num_params(model, print_out=True):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    if print_out:
        print('Trainable Parameters: %.3fM' % parameters)


# NOTE: you need this func to generate our sphinx doc
def get_parser():
    """Get parser of training arguments."""
    parser = configargparse.ArgumentParser(
        description='Train a new text-to-speech (TTS) model on one CPU, one or multiple GPUs',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--config', type=Path, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=Path, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model for logging, saving checkpoint")
    parser.add_argument('--outdir', type=Path, required=True,
                        help='Output directory')

    return parser


def main(cmd_args):
    """Run training."""
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)

    args = parser.parse_args(cmd_args)

    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    # logging info
    os.makedirs(hp.train.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(hp.train.log_dir,
                                             '%s-%d.log' % (args.name, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    # If --ngpu is not given,
    #   1. if CUDA_VISIBLE_DEVICES is set, all visible devices
    #   2. if nvidia-smi exists, use all devices
    #   3. else ngpu=0
    ngpu = hp.train.ngpu
    logger.info(f"ngpu: {ngpu}")

    # set random seed
    logger.info('random seed = %d' % hp.train.seed)
    random.seed(hp.train.seed)
    np.random.seed(hp.train.seed)

    vocoder = torch.hub.load('seungwonpark/melgan', 'melgan')  # load the vocoder for validation

    train(args, hp, hp_str, logger, vocoder)


if __name__ == "__main__":
    main(sys.argv[1:])
