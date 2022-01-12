import os
import argparse
import numpy as np

from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.utils import save_image


class ImageCropper:
    """
    Class to crop images
    """
    def get_random_crop(self, image, crop_height, crop_width):
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
        self, data_folder, dest_folder, size=(256, 256), samples=1
    ):
        """Crop image dataset

        :param data_folder: path of the folder with images
        :type data_folder: str
        :param dest_folder: path of the destination folder
        :type dest_folder: str
        :param size: cropping size, defaults to (256, 256)
        :type size: tuple, optional
        :param samples: number of samples per images, defaults to 1
        :type samples: int, optional
        """
        Path(dest_folder).mkdir(parents=True, exist_ok=True)
        for img_name in os.listdir(data_folder):
            image = Image.open(f"{data_folder}/{img_name}").convert("RGB")
            image = transforms.ToTensor()(image)
            for num in range(samples):
                image_cropped = self.get_random_crop(image, size[0], size[1])
                save_image(image_cropped, f"{dest_folder}/{num}_{img_name}")


def parse_arguments():
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
        "--size", help="size of the image cropping", default=None, nargs="+", type=int,
    )
    parser.add_argument(
        "--samples", help="number of random samples per image", default=1, type=int,
    )
    return parser.parse_args()


if __name__ == "__main__":
    cmd_args = parse_arguments()
    image_cropper = ImageCropper()
    image_cropper.create_cropped_dataset(
        cmd_args.data_folder, cmd_args.dest_folder, cmd_args.size, cmd_args.samples,
    )
