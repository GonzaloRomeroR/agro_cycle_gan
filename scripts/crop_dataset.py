import argparse
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


class ImageCropper:
    """
    Class to crop images
    """

    def get_random_crop(
        self, image: torch.Tensor, crop_height: int, crop_width: int
    ) -> torch.Tensor:
        """Randomly crop image

        :param image: image tensor
        :type image: ``Tensor`
        :param crop_height: height of the cropping
        :type crop_height: int
        :param crop_width: width of the cropping
        :type crop_width: int
        :return: cropped image tensor
        :rtype: `Tensor`
        """
        max_x = image.shape[2] - crop_width
        max_y = image.shape[1] - crop_height
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        crop = image[:, y : y + crop_height, x : x + crop_width]
        return crop

    def create_cropped_dataset(
        self,
        data_folder: str,
        dest_folder: str,
        crop_size: Tuple[int, ...] = (256, 256),
        resize: Optional[Tuple[int, ...]] = None,
        initial_resize: Optional[Tuple[int, ...]] = None,
        samples: int = 1,
    ) -> None:
        """Crop image dataset

        :param data_folder: path of the folder with images
        :type data_folder: str
        :param dest_folder: path of the destination folder
        :type dest_folder: str
        :param crop_size: cropping size, defaults to (256, 256)
        :type crop_size: tuple, optional
        :param resize: resizing after cropping, defaults to None
        :type resize: tuple, optional
        :param initial_resize: resize of the image before cropping, defaults to None
        :type resize: tuple, optional
        :param samples: number of samples per images, defaults to 1
        :type samples: int, optional
        """
        Path(dest_folder).mkdir(parents=True, exist_ok=True)
        for img_name in os.listdir(data_folder):
            image = Image.open(f"{data_folder}/{img_name}").convert("RGB")
            image = transforms.ToTensor()(image)
            if initial_resize:
                image = transforms.Resize(initial_resize[::-1])(image)
            for num in range(samples):
                image_cropped = self.get_random_crop(image, crop_size[0], crop_size[1])
                if resize:
                    image_cropped = transforms.Resize(resize[::-1])(image_cropped)
                save_image(image_cropped, f"{dest_folder}/{num}_{img_name}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop images in dataset")

    parser.add_argument(
        "data_folder",
        type=str,
        help="path to the folder containing images to crop",
        default=None,
    )
    parser.add_argument(
        "dest_folder",
        type=str,
        help="path to the folder where the cropped images will be stored",
        default=None,
    )
    parser.add_argument(
        "--crop_size",
        help="size of the image cropping",
        default=None,
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--resize",
        help="final resizing after cropping",
        default=None,
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--initial_resize",
        help="resize of image before cropping",
        default=None,
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--samples",
        help="number of random samples per image",
        default=1,
        type=int,
    )
    return parser.parse_args()


if __name__ == "__main__":
    cmd_args = parse_arguments()
    image_cropper = ImageCropper()
    image_cropper.create_cropped_dataset(
        cmd_args.data_folder,
        cmd_args.dest_folder,
        cmd_args.crop_size,
        cmd_args.resize,
        cmd_args.initial_resize,
        cmd_args.samples,
    )
