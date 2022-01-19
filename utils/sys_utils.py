import os

import nvidia_smi
import psutil
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


def suppress_tf_warnings():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def suppress_sklearn_errors():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_memory_usage():
    percentage = psutil.virtual_memory()[2]
    return percentage


def get_cpu_usage():
    cpu_usage = psutil.cpu_percent(4)
    return cpu_usage


def get_gpu_usage():
    try:
        nvidia_smi.nvmlInit()
        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        gpu_usage = {}
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            gpu_usage[nvidia_smi.nvmlDeviceGetName(handle)] = (
                100 * (info.total - info.free) / info.total
            )

        return gpu_usage
    except:
        print("WARNING: Cannot get gpu usage")
        return None

