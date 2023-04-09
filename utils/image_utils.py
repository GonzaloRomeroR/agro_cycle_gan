import os
from typing import Any, Tuple

import numpy as np
import torch
import pathlib
import torchvision.datasets as dset
import torchvision.transforms as transforms
from numpy.typing import NDArray
from PIL import Image
from dataclasses import dataclass


@dataclass
class DatasetDataloaders:
    name: str = ""
    train_A: torch.utils.data.DataLoader[Any] = None
    train_B: torch.utils.data.DataLoader[Any] = None
    test_A: torch.utils.data.DataLoader[Any] = None
    test_B: torch.utils.data.DataLoader[Any] = None


def datasets_get(
    dataset_name: str, im_size: Tuple[int, ...] = (64, 64), batch_size: int = 5
) -> DatasetDataloaders:
    """Upload and get the datasets for train and test

    :param dataset_name: name of the dataset to upload
    :type dataset_name: str
    :param im_size: image size
    :type im_size: tuple (h, w), optional
    :param batch_size: number of images per batch
    :type batch_size: int, optinal
    :return: DatasetDataloaders with the images of the two domains
    :rtype: DatasetDataloaders
    """

    datasets = DatasetDataloaders()
    datasets.name = dataset_name
    (datasets.train_A, datasets.train_B) = get_datasets(
        dataset_name=dataset_name,
        dataset="train",
        im_size=im_size,
        batch_size=batch_size,
    )
    (datasets.test_A, datasets.test_B) = get_datasets(
        dataset_name=dataset_name,
        dataset="test",
        im_size=im_size,
        batch_size=batch_size,
    )
    return datasets


def get_datasets(
    dataset_name: str,
    dataset: str = "train",
    im_size: Tuple[int, ...] = (64, 64),
    batch_size: int = 5,
) -> Tuple[torch.utils.data.DataLoader[Any], ...]:
    """Upload and get the datasets

    :param dataset_name: name of the dataset to upload
    :type dataset_name: str
    :param dataset: type of the dataset, test or train, defaults to "train"
    :type dataset: str, optional
    :return: DataLoader tuple with the images of the two domains
    :rtype: ´DataLoader´ tuple
    """
    file_path = pathlib.Path(__file__).parent.resolve()
    images_A = upload_images(
        path=f"{file_path}/../images/{dataset_name}/{dataset}_A/",
        im_size=im_size,
        batch_size=batch_size,
    )
    images_B = upload_images(
        path=f"{file_path}/../images/{dataset_name}/{dataset}_B/",
        im_size=im_size,
        batch_size=batch_size,
    )
    return images_A, images_B


def upload_images(
    path: str, im_size: Tuple[int, ...], batch_size: int = 5, num_workers: int = 2, data_augmentation: bool = True
) -> torch.utils.data.DataLoader[Any]:
    """Upload images from folder

    :param path: path to the folder with images
    :type path: str
    :param im_size: size of the image
    :type im_size: tuple
    :param batch_size: batch size, defaults to 5
    :type batch_size: int, optional
    :param num_workers: number of workers processes, defaults to 2
    :type num_workers: int, optional
    :return: image dataset
    :rtype: `DataLoader`
    """
    if data_augmentation:
        image_dataset = dset.ImageFolder(
            root=path,
            transform=transforms.Compose(
                [
                    transforms.RandomResizedCrop(im_size, scale=(0.6, 1)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=(-5, 5)),
                    transforms.RandomAdjustSharpness(sharpness_factor=2),
                    transforms.RandomAutocontrast(),
                    transforms.ToTensor(),
                    transforms.Normalize((0, 0, 0), (1, 1, 1)),
                ]
            ),
        )

    else:
        image_dataset = dset.ImageFolder(
            root=path,
            transform=transforms.Compose(
                [
                    transforms.Resize(im_size),
                    transforms.CenterCrop(im_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0, 0, 0), (1, 1, 1)),
                ]
            ),
        )

    images = torch.utils.data.DataLoader(
        dataset=image_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    return images


def upload_images_numpy(path: str, im_size: Tuple[int, ...]) -> NDArray[Any]:
    """Upload images from folder

    :param path: path to the folder with images
    :type path: str
    :param im_size: size of the image
    :type im_size: tuple
    :return: image dataset
    :rtype: numpy array
    """
    img_list = []
    for file_name in os.listdir(path):
        image = Image.open(f"{path}/{file_name}")
        image = image.resize(im_size)
        img_list.append(np.asarray(image))
        image.close()
    return np.array(img_list)
