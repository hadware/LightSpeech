import argparse

import onnxruntime

import numpy as np

ONNX_CPU_PROVIDERS = [
    "CPUExecutionProvider",
]

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str,
                    default="lightspeech.onnx",)

if __name__ == '__main__':
    args = parser.parse_args()

    session = onnxruntime.InferenceSession(args.model, providers=ONNX_CPU_PROVIDERS)

    dummy_input = {
        "x": np.array([27, 45, 67, 45, 82, 9, 69, 30, 67, 69, 84, 84]),
    }


    out = session.run(None, dummy_input)
    print(out[0].shape)