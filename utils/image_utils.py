import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
import numpy as np

from PIL import Image


def get_datasets(dataset_name, dataset="train", im_size=(64, 64), batch_size=5):
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


def upload_images(path, im_size, batch_size=5, num_workers=2):
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


def upload_images_numpy(path, im_size):
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
