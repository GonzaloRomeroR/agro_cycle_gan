import os
from typing import Any, Dict, Optional

import nvidia_smi
import psutil
import torch


def system_configuration() -> None:
    suppress_sklearn_errors()
    suppress_qt_warnings()


def get_device(debug: bool = False) -> torch.device:
    if debug:
        if torch.cuda.is_available():
            print("The code will run on GPU.")
        else:
            print("The code will run on CPU.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def suppress_qt_warnings() -> None:
    os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"


def suppress_tf_warnings() -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def suppress_sklearn_errors() -> None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_memory_usage() -> Any:
    percentage = psutil.virtual_memory()[2]
    return percentage


def get_cpu_usage() -> Any:
    cpu_usage = psutil.cpu_percent(4)
    return cpu_usage


def get_gpu_usage() -> Optional[Dict[Any, Any]]:
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
