import torch
import gdown
import sys
import os
import shutil
import zipfile

from models.model_factory import ModelCreator
from pathlib import Path


def create_models(device, **kwargs):
    """Create discriminator and generator models

    :param device: pytorch device to be used
    :type device: str
    :return: tuple with the generators and discriminators
    :rtype: tuple
    """
    model_creator = ModelCreator()
    D_A = model_creator.create(model_type="disc", model_name="", **kwargs).to(device)
    D_B = model_creator.create(model_type="disc", model_name="", **kwargs).to(device)
    G_A2B = model_creator.create(model_type="gen", model_name="", **kwargs).to(device)
    G_B2A = model_creator.create(model_type="gen", model_name="", **kwargs).to(device)
    return G_A2B, G_B2A, D_A, D_B


def save_models(path, name, G_A2B, G_B2A, D_A, D_B):
    """Save trained models

    :param G_A2B: Generator to transform from A to B 
    :type G_A2B: Generator
    :param G_B2A: generator to transform from B to A
    :type G_B2A: Generator
    :param D_A: discriminator for A images
    :type D_A: Discriminator
    :param D_B: discriminator for B images
    :type D_B: Discriminator
    :param path: path to the folder to store de models
    :type path: str
    :param name: name of the file to save
    :type name: [type]
    """
    Path(path).mkdir(parents=True, exist_ok=True)

    torch.save(G_A2B, path + "/" + name + "_G_A2B.pt")
    torch.save(G_B2A, path + "/" + name + "_G_B2A.pt")
    torch.save(D_A, path + "/" + name + "_D_A.pt")
    torch.save(D_B, path + "/" + name + "_D_B.pt")


def load_models(path, name):
    """Load trained models

    :return: tuple with the loaded models
    :rtype: tuple
    """
    G_A2B = torch.load(path + "/" + name + "_G_A2B.pt")
    G_B2A = torch.load(path + "/" + name + "_G_B2A.pt")
    D_A = torch.load(path + "/" + name + "_D_A.pt")
    D_B = torch.load(path + "/" + name + "_D_B.pt")
    return G_A2B, G_B2A, D_A, D_B


def load_generators(path, name):
    """Load generator models

    :return: tuple with the loaded models
    :rtype: tuple
    """
    G_A2B = torch.load(path + "/" + name + "_G_A2B.pt")
    G_B2A = torch.load(path + "/" + name + "_G_B2A.pt")
    return G_A2B, G_B2A


def download_images(image_type, path="images/"):
    """Download images from and url

    :param image_type: type of the dataset to download
    :type image_type: str
    :param path: path to store the images
    :type path: str
    """
    if image_type == "horse2zebra":
        download_zebras(path)
    else:
        raise RuntimeError("Dataset not found")


def download_zebras(path):
    zip_name = "horse2zebra.zip"
    zip_path = path + zip_name
    if not os.path.exists(zip_path.rsplit(".", 1)[0]):
        url = "https://drive.google.com/uc?id=1jPelB2jzNZJq3ZU9Uk_Mkt4MJtF3DRgg"
        gdown.download(url, path + zip_name, quiet=False)
        unzip(zip_path)
        # Move files
        shutil.move(f"{path}/horse2zebra/train/A", f"{path}/horse2zebra/train_A/A")
        shutil.move(f"{path}/horse2zebra/train/B", f"{path}/horse2zebra/train_B/B")
        shutil.move(f"{path}/horse2zebra/test/A", f"{path}/horse2zebra/test_A/A")
        shutil.move(f"{path}/horse2zebra/test/B", f"{path}/horse2zebra/test_B/B")
    else:
        print("Dataset is already downloaded")


def unzip(path):
    """Unzips file

    :param path: path to the file to unzip
    :type path: str
    """
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(path.rsplit("/", 1)[0])


def move_files():
    shutil.move("/content/horse2zebra/train/A", "/content/horses_train/A")
    shutil.move("/content/horse2zebra/train/B", "/content/zebra_train/B")
    shutil.move("/content/horse2zebra/test/A", "/content/horses_test/A")
    shutil.move("/content/horse2zebra/test/B", "/content/zebra_test/B")

