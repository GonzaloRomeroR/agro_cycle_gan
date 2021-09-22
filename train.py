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
    D_A = model_creator.create(model_type="disc", model_name="").to(device)
    D_B = model_creator.create(model_type="disc", model_name="").to(device)
    G_A2B = model_creator.create(model_type="gen", model_name="").to(device)
    G_B2A = model_creator.create(model_type="gen", model_name="").to(device)

    # save_models(f"./results/{dataset_name}", dataset_name, G_A2B, G_B2A, D_A, D_B)
    # load_models(f"./results/{dataset_name}", dataset_name)


def train(G_A2B, G_B2A, D_A, D_B, path, images_A, images_B, name, num_epochs=20):
    """Train the generator and the discriminator

    :param G_A2B: generator to transform images from A to B
    :type G_A2B: `Generator`
    :param G_B2A: generator to transform images from B to A
    :type G_B2A: `Generator`
    :param D_A: discriminator for images in the domain A
    :type D_A: `Discriminator`
    :param D_B: discriminator for images in the domain A
    :type D_B: `Discriminator`
    :param images_A: dataloader with images of the domain A
    :type images_A: ´DataLoader´
    :param images_B: dataloader with images of the domain B
    :type images_B: ´DataLoader´
    :param path: path to load and save the models
    :type path: str
    :param name: name of the models
    :type name: str
    :param num_epochs: number of epochs, defaults to 20
    :type num_epochs: int, optional
    """
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
