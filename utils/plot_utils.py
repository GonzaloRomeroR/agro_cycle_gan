import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import torch


def show_batch(batch, device):
    """Show image batch

    :param batch: batch of images to plot  
    :type batch: list
    :param device: pytorch device  
    :type device: pytorch device
    """
    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.title("Batch images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                batch[0].to(device)[:10], padding=2, normalize=True,
            ).cpu(),
            (1, 2, 0),
        )
    )
    plt.show()

