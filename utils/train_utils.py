from typing import Tuple
from dataclasses import dataclass, field

import torch
from models.generators.base_generator import BaseGenerator


@dataclass
class Config:
    use_dataset: str = None
    download_dataset: bool = False
    image_resize: list = field(default_factory=list)
    tensorboard: bool = False
    store_models: bool = False
    load_models: bool = False
    batch_size: int = 5
    num_epochs: int = 1
    metrics: bool = True
    generator: str = ""
    discriminator: str = ""
    metrics: bool = False


def generate_images_cycle(
    a_real: torch.Tensor,
    b_real: torch.Tensor,
    G_A2B: BaseGenerator,
    G_B2A: BaseGenerator,
) -> Tuple[torch.Tensor, ...]:
    """
    Create fake and reconstructed images.
    """
    b_fake = G_A2B(a_real)
    a_recon = G_B2A(b_fake)
    a_fake = G_B2A(b_real)
    b_recon = G_A2B(a_fake)
    return a_fake, b_fake, a_recon, b_recon
