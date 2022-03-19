from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from models.generators.base_generator import BaseGenerator


def show_batch(
    batch: List[Any], device: torch.device, title: str = "Batch images"
) -> None:
    """Show image batch

    :param batch: batch of images to plot  
    :type batch: list
    :param device: pytorch device  
    :type device: pytorch device
    """
    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.title(title)
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                batch[0].to(device)[:10], padding=2, normalize=True,
            ).cpu(),
            (1, 2, 0),
        )
    )
    plt.show()


def plot_generator_images(
    G_A2B: BaseGenerator,
    G_B2A: BaseGenerator,
    dataloader_A: torch.utils.data.DataLoader[Any],
    dataloader_B: torch.utils.data.DataLoader[Any],
    device: torch.device,
) -> None:
    """Plot generated images

    :param G_A2B: generator to transform from A to B
    :type G_A2B: `Generator`
    :param G_B2A: generator to transform from B to A
    :type G_B2A: `Generator`
    :param dataloader_A: dataloader with images of the domain A
    :type dataloader_A: `Dataloader`
    :param dataloader_B: dataloader with images of the domain B
    :type dataloader_B: `Dataloader`
    :param device: pytorch  
    :type device: `Device`
    """
    batch_a_test = next(iter(dataloader_A))[0].to(device)
    real_a_test = batch_a_test.cpu().detach()
    fake_b_test = G_A2B(batch_a_test).cpu().detach()

    show_batch(real_a_test, device, title="Real A")
    show_batch(fake_b_test, device, title="Fake B")

    batch_b_test = next(iter(dataloader_B))[0].to(device)
    real_b_test = batch_b_test.cpu().detach()
    fake_a_test = G_B2A(batch_b_test).cpu().detach()

    show_batch(real_b_test, device, title="Real B")
    show_batch(fake_a_test, device, title="Fake A")

