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
from lightspeech.optispeech import OptiSpeechGenerator
from lightspeech.utils.display import num_params
from lightspeech.utils.hparams import HParam
from lightspeech.utils.plot import generate_audio, plot_spectrogram_to_numpy, plot_time_series_to_numpy
from lightspeech.utils.util import read_wav_np, get_commit_hash

BATCH_COUNT_CHOICES = ["auto", "seq", "bin", "frame"]
BATCH_SORT_KEY_CHOICES = ["input", "output", "shuffle"]


def load_model(config_path: str, config_name: str) -> OptiSpeechGenerator:
    # Charger la configuration
    config_path = Path(config_path).absolute()
    with hydra.initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = hydra.compose(config_name=config_name)
        # Instancier le modèle
        model = hydra.utils.instantiate(cfg)
        return model


def train(args, hp, hp_str, logger, vocoder):
    chckpt_dir: Path = Path(hp.train.chkpt_dir)
    outdir: Path = args.outdir
    assets_dir: Path = outdir / 'assets'
    data_dir: Path = hp.data.data_dir
    ckckpt_path: Path = args.checkpoint_path
    log_dir: Path = Path(hp.train.log_dir)

    # makedir all previous folders
    chckpt_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if hp.train.ngpu > 0 else "cpu")

    dataloader = loader.get_tts_dataset(data_dir, hp.train.batch_size, hp)
    validloader = loader.get_tts_dataset(data_dir, 5, hp, True)

    out_dim = hp.audio.num_mels
    model_partial = load_model(hp.model.config_folder, hp.model.config_name)
    model: OptiSpeechGenerator = model_partial(out_dim)
    # set torch device
    model = model.to(device)
    print("Model is loaded ...")
    print("New Training")
    global_step = 0
    optimizer = get_std_opt(model, model.hidden_dim, hp.model.transformer_warmup_steps, hp.model.transformer_lr)

    print("Batch Size :", hp.train.batch_size)

    num_params(model)
    num_params(vocoder)

    log_path = log_dir / args.name
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_path))
    model.train()
    forward_count = 0
    githash = get_commit_hash()

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
            loss, report_dict, preds_dict, _ = model(
                x=x.cuda(),
                x_lengths=input_length.cuda(),
                mel=y.cuda(),
                mel_lengths=out_length.cuda(),
                gt_durations=dur.cuda(),
                energies=e.cuda(),
                pitches=p.cuda()
            )
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

                for k, v in report_dict.items():
                    if k is not None and v is not None:
                        writer.add_scalar("main/{}".format(k), v, step)

            if step % hp.train.validation_step == 0:

                for valid in validloader:
                    x_, input_length_, y_, _, out_length_, ids_, dur_, e_, p_ = valid
                    model.eval()
                    with torch.no_grad():
                        loss_, report_dict_, preds_dict_, debug_data = model(
                            x=x_.cuda(),
                            x_lengths=input_length_.cuda(),
                            mel=y_.cuda(),
                            mel_lengths=out_length_.cuda(),
                            gt_durations=dur_.cuda(),
                            energies=e_.cuda(),
                            pitches=p_.cuda()
                        )

                        synth_results = model.synthesise_one(x_[-1].cuda())  # [T, num_mel]
                        mels_ = synth_results['mels'][0]
                    model.train()
                    for k, v in report_dict_.items():
                        if k is not None and v is not None:
                            writer.add_scalar("validation/{}".format(k), v, step)
                    break

                mels_ = mels_.T  # Out: [num_mels, T]
                writer.add_image('melspectrogram_target',
                                 plot_spectrogram_to_numpy(y_[-1].T.data.cpu().numpy()[:, :out_length_[-1]]),
                                 step, dataformats='HWC')
                writer.add_image('melspectrogram_prediction',
                                 plot_spectrogram_to_numpy(mels_.data.cpu().numpy()),
                                 step, dataformats='HWC')
                writer.add_image('before_decoding',
                                 plot_spectrogram_to_numpy(debug_data["before_decoding"][-1].numpy().T[:, :out_length_[-1]]),
                                 step, dataformats='HWC')
                writer.add_image('pitch_prediction_inference',
                                 plot_time_series_to_numpy(synth_results["pitch"][0].numpy()),
                                 step, dataformats='HWC')
                writer.add_image('pitch_prediction_forward',
                                 plot_time_series_to_numpy(preds_dict_["pitch"][-1].numpy()[:input_length_[-1]]),
                                 step, dataformats='HWC')
                writer.add_image('pitch_target_avg',
                                 plot_time_series_to_numpy(debug_data["avg_pitch"][-1].numpy()[:input_length_[-1]]),
                                 step, dataformats='HWC')
                writer.add_image('pitch_target',
                                 plot_time_series_to_numpy(p_[-1].numpy()[:out_length_[-1]]),
                                 step, dataformats='HWC')
                writer.add_image('energy_prediction',
                                 plot_time_series_to_numpy(synth_results["energy"][0].numpy()),
                                 step, dataformats='HWC')
                writer.add_image('duration_prediction',
                                 plot_time_series_to_numpy(synth_results["durations"][0].numpy()),
                                 step, dataformats='HWC')

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
                save_path = chckpt_dir / args.name / '{}_fastspeech_{}_{}k_steps.pyt'.format(args.name, githash,
                                                                                             step // 1000)
                save_path.parent.mkdir(parents=True, exist_ok=True)

                torch.save({
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'step': step,
                    'hp_str': hp_str,
                    'githash': githash,
                }, save_path)
                logger.info("Saved checkpoint to: %s" % save_path)
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))



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
