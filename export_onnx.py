from lightspeech import FeedForwardTransformer
from utils.hparams import HParam
from dataset.texts import valid_symbols
import configargparse
import torch
import sys

DEFAULT_OPSET = 16
DEFAULT_SEED = 4577

parser = configargparse.ArgumentParser(
    description='Train a new text-to-speech (TTS) model on one CPU, one or multiple GPUs',
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-c', '--config', type=str, required=True,
                    help="yaml file for configuration")
parser.add_argument('-n', '--name', type=str, required=True,
                    help="name of the model for logging, saving checkpoint")
parser.add_argument('--outdir', type=str, required=True,
                    help='Output directory')
parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version to use (default 15")
parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
parser.add_argument('-t', '--trace', action='store_true', help="For JIT Trace Module")



def main(cmd_args):

    args = parser.parse_args(cmd_args)

    hp = HParam(args.config)

    dummy_input = {
        "x": torch.randint(0, len(valid_symbols), size=(50,)),
    }

    input_names = list(dummy_input.keys())
    output_names = ["wav"]

    dynamic_axes = {
        "x": {0: "time"},
        "wav": {0: "frames"},
    }

    idim = len(valid_symbols)
    odim = hp.audio.num_mels
    model = FeedForwardTransformer(idim, odim, hp)
    model.forward = model.inference
    torch.onnx.export(
        model,
        f="lightspeech.onnx",
        args=tuple(dummy_input.values()),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        # export_params=True,
        do_constant_folding=True,
    )


if __name__ == "__main__":
    main(sys.argv[1:])

