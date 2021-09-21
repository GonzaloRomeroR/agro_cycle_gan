import torch
import argparse
import os

from utils.file_utils import download_images, load_models, save_models
from utils.image_utils import get_datasets
from utils.plot_utils import show_batch

from models.model_factory import ModelCreator


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def setup(cmd_args):
    print("Performing setup")
    device = get_device()
    if dataset_name := cmd_args.download_dataset:
        print(f"Downloading dataset {dataset_name}")
        download_images(dataset_name)

    train_images_A, train_images_B = get_datasets(
        dataset_name=dataset_name, dataset="train"
    )
    model_creator = ModelCreator()
    D = model_creator.create(model_type="disc", model_name="")
    G = model_creator.create(model_type="gen", model_name="")


def train():
    pass


def get_device():
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download_dataset",
        type=str,
        help="name of the dataset to download",
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    cmd_args = parse_arguments()
    setup(cmd_args)
