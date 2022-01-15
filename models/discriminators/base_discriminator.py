import torch

from ..base_model import BaseModel
from abc import abstractmethod


class BaseDiscriminator(BaseModel):
    """
    Base Discriminator class
    """

    def _set_params(self, ndf=64, nc=3, lr=0.0002, beta_1=0.5):
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
    def _create_model(self):
        pass

    def _set_optimizer(self):
        self.optimizer = self.get_optimizer()

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(self.beta_1, 0.999)
        )

    def get_loss(self, real, fake):
        return torch.mean((real - 1) ** 2) + torch.mean(fake ** 2)

    def forward(self, input):
        return self.main(input)
