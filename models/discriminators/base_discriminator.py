from abc import abstractmethod

import torch

from ..base_model import BaseModel
from typing import Any


class BaseDiscriminator(BaseModel):
    """
    Base Discriminator class
    """

    def _set_params(self, ndf: int = 64, nc: int = 3, lr: float = 0.0002, beta_1: float = 0.5) -> None:
        """Set discriminator parameters

        :param ndf: size of feature maps in discriminator, defaults to 64
        :type ndf: int, optional
        :param nc: number of channels in the training images, defaults to 3
        :type nc: int, optional
        :param lr: learning rate, defaults to 0.0002
        :type lr: float, optional
        :param beta_1: beta_1 for Adam optimizer, defaults to 0.5
        :type beta_1: float, optional
        """
        self.nc = nc
        self.ndf = ndf
        self.lr = lr
        self.beta_1 = beta_1

    @abstractmethod
    def _create_model(self) -> None:
        pass

    def _set_optimizer(self) -> None:
        self.optimizer = self.get_optimizer()

    def get_optimizer(self) -> Any:
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(self.beta_1, 0.999)
        )

    def get_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        return torch.mean((real - 1) ** 2) + torch.mean(fake ** 2)

    def forward(self, input: torch.Tensor) -> Any:
        return self.main(input)
