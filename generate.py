import argparse
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from typing import Optional

from utils.file_utils import load_generators


class ImageTransformer:
    """
    Class to transform datasets between domains
    """

    def __init__(self, dataset_name: str):
        """Constructor for the ImageTransformer

        :param dataset_name: name of the dataset to load the trained generators,
        the folder with the models will be searched in the results folder
        :type dataset_name: str
        """
        self.dataset_name = dataset_name
        self.G_A2B, self.G_B2A = load_generators(
            f"./results/{dataset_name}", dataset_name
        )
        Path(f"./images_gen/{dataset_name}").mkdir(parents=True, exist_ok=True)

    def transform_image(self, image: torch.Tensor, domain: str = "B") -> Any:
        """Transforms images from one domain to another

        :param image: image tensor to transform
        :type image: Tersor
        :param domain: domain for the images to be transformed, defaults to "B"
        :type domain: str, optional
        :return: transformed image tensor
        :rtype: Tensor
        """
        if domain == "A":
            self.G_B2A.eval()
            return self.G_B2A(image)
        elif domain == "B":
            self.G_A2B.eval()
            return self.G_A2B(image)
        else:
            raise ValueError("Domain is not valid")

    def transform_dataset(
        self,
        origin_path: str,
        dest_path: str,
        domain: str = "B",
        resize: Any = None,
        image_num: Optional[int] = None,
    ) -> None:
        """Transforms images folders from one domain to another

        :param origin_path: path of the folder with the images to transform
        :type origin_path: str
        :param dest_path: path of the folder for the transform images to be stored
        :type dest_path: str
        :param domain: domain for the images to be transformed, defaults to "A"
        :type domain: str, optional
        :param resize: tuple to resize the input, defaults to None
        :type resize: tuple, optional
        :param image_num: number of images to generate, defaults to None
        :type resize: int, optional
        """
        print(f"Tranforming dataset")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        if image_num is None:
            image_num = len(os.listdir(origin_path))
        for i, img_name in enumerate(os.listdir(origin_path)):
            if i == image_num:
                break

            if not img_name.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
            ):
                # Not a valid image
                continue

            image = Image.open(f"{origin_path}/{img_name}").convert("RGB")
            if resize:
                image = transforms.Resize(resize)(image)
            image = transforms.ToTensor()(image).to(device)
            image = torch.unsqueeze(image, dim=0)
            image_trans = self.transform_image(image, domain)
            save_image(image_trans, f"{dest_path}/{img_name}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform image from one domain to another"
    )
    parser.add_argument(
        "images_path",
        type=str,
        help="path to the folder with the images to transform",
        default=None,
    )
    parser.add_argument(
        "dest_path",
        type=str,
        help="path to the folder to store the transformed images",
        default=None,
    )
    parser.add_argument(
        "--generator_name",
        type=str,
        help="name of the generators to use",
        default=None,
    )
    parser.add_argument(
        "--dest_domain",
        type=str,
        help="domain of the tranformed images, A or B",
        default="B",
    )
    parser.add_argument(
        "--image_resize",
        help="size of the image to resize",
        default=None,
        nargs=2,
        type=int,
    )
    return parser.parse_args()


if __name__ == "__main__":
    cmd_args = parse_arguments()
    image_transformer = ImageTransformer(cmd_args.generator_name)
    image_transformer.transform_dataset(
        cmd_args.images_path,
        cmd_args.dest_path,
        cmd_args.dest_domain,
        cmd_args.image_resize,
    )
