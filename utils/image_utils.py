import os
from typing import Any, Tuple

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image


def datasets_get(
    dataset_name: str, im_size: Tuple[int, int] = (64, 64), batch_size: int = 5
) -> Tuple[torch.utils.data.DataLoader[Any], ...]:
    """Upload and get the datasets for train and test

    :param dataset_name: name of the dataset to upload
    :type dataset_name: str
    :param im_size: image size
    :type im_size: tuple (h, w), optional
    :param batch_size: number of images per batch
    :type batch_size: int, optinal
    :return: DataLoader tuple with the images of the two domains
    :rtype: ´DataLoader´ tuple
    """
    train_A, train_B = get_datasets(
        dataset_name=dataset_name,
        dataset="train",
        im_size=im_size,
        batch_size=batch_size,
    )
    test_A, test_B = get_datasets(
        dataset_name=dataset_name,
        dataset="test",
        im_size=im_size,
        batch_size=batch_size,
    )
    return train_A, train_B, test_A, test_B


def get_datasets(
    dataset_name: str,
    dataset: str = "train",
    im_size: Tuple[int, int] = (64, 64),
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
    images_A = upload_images(
        path=f"./images/{dataset_name}/{dataset}_A/",
        im_size=im_size,
        batch_size=batch_size,
    )
    images_B = upload_images(
        path=f"./images/{dataset_name}/{dataset}_B/",
        im_size=im_size,
        batch_size=batch_size,
    )
    return images_A, images_B


def upload_images(
    path: str, im_size: Tuple[int, int], batch_size: int = 5, num_workers: int = 2
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


def upload_images_numpy(path: str, im_size: Tuple[int, int]) -> np.ndarray:
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
    return np.array(img_list)
