import torch
import argparse
import os
import random

from utils.file_utils import load_generators


class ImageTransformer:
    def __init__(self, dataset_name):
        self.G_A2B, self.G_B2A = load_generators(
            f"./results/{dataset_name}", dataset_name
        )

    def transform_image(self):
        pass

    def transform_dataset(self):
        pass


def parse_arguments():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


if __name__ == "__main__":
    cmd_args = parse_arguments()
    image_transformer = ImageTransformer("horse2zebra")
    image_transformer.transform_image()
