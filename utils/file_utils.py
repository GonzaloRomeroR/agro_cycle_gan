import os
import shutil
import sys
import zipfile

from pathlib import Path
from typing import Any, Dict, Tuple


try:
    import kaggle
except IOError:
    print("Cannot import kaggle, could not find kaggle.json")

import gdown
import torch
from models.discriminators.base_discriminator import BaseDiscriminator
from models.generators.base_generator import BaseGenerator
from models.model_factory import ModelCreator

from .sys_utils import get_device


def get_models(
    dataset_name: str,
    device: torch.device,
    load: bool = False,
    disc_name: str = "",
    gen_name: str = "",
) -> Tuple[BaseGenerator, BaseGenerator, BaseDiscriminator, BaseDiscriminator]:
    """Obtain Generator and Discriminator models

    :param dataset_name: name of the dataset
    :type dataset_name: str
    :param device: type of device
    :type device: str
    :param load: flag to load the models from previous runs, defaults to False
    :type load: bool, optional
    :return: tuple with the models obtained
    :rtype: tuple of `Models`
    """
    if load:
        G_A2B, G_B2A, D_A, D_B = load_models(f"./results/{dataset_name}", dataset_name)
    else:
        G_A2B, G_B2A, D_A, D_B = create_models(
            device, disc_name=disc_name, gen_name=gen_name
        )
    return G_A2B, G_B2A, D_A, D_B


def create_models(
    device: torch.device,
    gen_name: str = "",
    disc_name: str = "",
    **kwargs: Dict[str, Any],
) -> Tuple[BaseGenerator, BaseGenerator, BaseDiscriminator, BaseDiscriminator]:
    """Create discriminator and generator models

    :param device: pytorch device to be used
    :type device: str
    :return: tuple with the generators and discriminators
    :rtype: tuple
    """
    model_creator = ModelCreator()
    D_A = model_creator.create(model_type="disc", model_name=disc_name, **kwargs).to(
        device
    )
    D_B = model_creator.create(model_type="disc", model_name=disc_name, **kwargs).to(
        device
    )
    G_A2B = model_creator.create(model_type="gen", model_name=gen_name, **kwargs).to(
        device
    )
    G_B2A = model_creator.create(model_type="gen", model_name=gen_name, **kwargs).to(
        device
    )
    return G_A2B, G_B2A, D_A, D_B


def save_models(
    path: str,
    name: str,
    G_A2B: BaseGenerator,
    G_B2A: BaseGenerator,
    D_A: BaseDiscriminator,
    D_B: BaseDiscriminator,
) -> None:
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


def load_models(
    path: str, name: str
) -> Tuple[BaseGenerator, BaseGenerator, BaseDiscriminator, BaseDiscriminator]:
    """Load trained models

    :return: tuple with the loaded models
    :rtype: tuple
    """
    try:
        G_A2B, G_B2A = load_generators(path, name)
        D_A, D_B = load_discriminators(path, name)
    except FileNotFoundError:
        print(f"Failed loading models: Cannot find trained models in {path}")
        exit(0)

    return G_A2B, G_B2A, D_A, D_B


def load_generators(path: str, name: str) -> Tuple[BaseGenerator, BaseGenerator]:
    """Load generator models

    :return: tuple with the loaded models
    :rtype: tuple
    """
    G_A2B = torch.load(path + "/" + name + "_G_A2B.pt", map_location=get_device())
    G_B2A = torch.load(path + "/" + name + "_G_B2A.pt", map_location=get_device())
    return G_A2B, G_B2A


def load_discriminators(
    path: str, name: str
) -> Tuple[BaseDiscriminator, BaseDiscriminator]:
    """Load discriminators models

    :return: tuple with the loaded models
    :rtype: tuple
    """
    D_A = torch.load(path + "/" + name + "_D_A.pt", map_location=get_device())
    D_B = torch.load(path + "/" + name + "_D_B.pt", map_location=get_device())
    return D_A, D_B


def download_images(image_type: str, path: str = "images/") -> None:
    """Download images from and url

    :param image_type: type of the dataset to download
    :type image_type: str
    :param path: path to store the images
    :type path: str
    """
    print(f"Downloading dataset {image_type}")
    if image_type == "horse2zebra":
        download_horse2zebras(path)
    elif image_type == "soy_small2soy_big":
        download_soy_small2soy_big(path)
    elif image_type == "soy2corn":
        download_soy2corn(path)
    elif image_type == "over_corn2over_wheat":
        download_over_corn2over_wheat(path)
    else:
        raise RuntimeError("Dataset not found")


def download_soy2corn(path: str) -> None:
    breakpoint()
    if not os.path.exists(path + "/soy2corn"):
        if "kaggle" not in sys.modules:
            raise RuntimeError(
                "Cannot download dataset since kaggle could not be imported"
            )
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files("gonzaromeror/soycorn", path=path, unzip=True)
    else:
        print("Dataset is already downloaded")


def download_soy_small2soy_big(path: str) -> None:
    if not os.path.exists(path + "/soy_small2soy_big"):
        if "kaggle" not in sys.modules:
            raise RuntimeError(
                "Cannot download dataset since kaggle could not be imported"
            )
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "gonzaromeror/soysmallbig", path=path, unzip=True
        )
    else:
        print("Dataset is already downloaded")


def download_over_corn2over_wheat(path: str) -> None:
    if not os.path.exists(path + "/over_corn2over_wheat"):
        if "kaggle" not in sys.modules:
            raise RuntimeError(
                "Cannot download dataset since kaggle could not be imported"
            )
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "gonzaromeror/overcornoverwheat", path=path, unzip=True
        )
    else:
        print("Dataset is already downloaded")


def download_horse2zebras(path: str) -> None:
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


def unzip(path: str) -> None:
    """Unzips file

    :param path: path to the file to unzip
    :type path: str
    """
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(path.rsplit("/", 1)[0])
