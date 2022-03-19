from typing import Tuple

import torch
from models.generators.base_generator import BaseGenerator


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
