import torch
import argparse
import os


from utils.file_utils import load_generators
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.utils import save_image


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

    def transform_image(self, image, domain="A"):
        """Transforms images from one domain to another

        :param image: image tensor to transform
        :type image: Tersor
        :param domain: domain for the images to be transformed, defaults to "A"
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
            raise (ValueError, "Domain is not valid")

    def transform_dataset(
        self, origin_path: str, dest_path: str, domain="A", resize=None
    ):
        """Transforms images folders from one domain to another

        :param origin_path: path of the folder with the images to transform
        :type origin_path: str
        :param dest_path: path of the folder for the transform images to be stored
        :type dest_path: str
        :param domain: domain for the images to be transformed, defaults to "A"
        :type domain: str, optional
        :param resize: tuple to resize the input, defaults to None
        :type resize: tuple, optional
        """
        print(f"Tranforming dataset")
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        for img_name in os.listdir(origin_path):
            image = Image.open(f"{origin_path}/{img_name}").convert("RGB")
            if resize:
                image = transforms.Resize(resize)(image)
            image = transforms.ToTensor()(image)
            image = torch.unsqueeze(image, dim=0)
            image_trans = self.transform_image(image, domain)
            save_image(image_trans, f"{dest_path}/{img_name}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator_name",
        type=str,
        help="name of the generators to use",
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    cmd_args = parse_arguments()
    image_transformer = ImageTransformer(cmd_args.generator_name)
    image_transformer.transform_dataset(
        "./images/horse2zebra/test_A/A", "./images_gen/horse2zebra/"
    )

