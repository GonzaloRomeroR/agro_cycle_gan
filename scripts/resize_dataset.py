#!/usr/bin/python
import os

from PIL import Image
from typing import Tuple


def resize_image_folder(path: str, size: Tuple[int, int]) -> None:
    """
    Resize folder of images to an specific size
    """
    dirs = os.listdir(path)
    for item in dirs:

        if not item.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
        ):
            continue

        print(f"Resizing {item} to {size}")
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            f, _ = os.path.splitext(path + item)
            imResize = im.resize(size, Image.ANTIALIAS)
            imResize.save(f + ".jpg", "JPEG", quality=90)


if __name__ == "__main__":
    path = ""
    size = (1920, 1080)
    resize_image_folder(path, size)
