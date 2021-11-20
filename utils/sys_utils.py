import os
import torch


def get_device(debug=False):
    if debug:
        if torch.cuda.is_available():
            print("The code will run on GPU.")
        else:
            print("The code will run on CPU.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def suppress_qt_warnings():
    os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"
