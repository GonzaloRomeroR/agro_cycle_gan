import torch

from ..base_model import BaseModel
from abc import abstractmethod


class BaseGenerator(BaseModel):
    """
    Base Generator class
    """

    def _set_params(self, ngf=64, nc=3, blocks=9, lr=0.0002, beta_1=0.5):
        """Set generator parameters

        :param ngf: number of channels in the training images, defaults to 64
        :type ngf: int, optional
        :param nc: size of feature maps in generator, defaults to 3
        :type nc: int, optional
        :param blocks: number of ResBlocks, defaults to 9
        :type blocks: int, optional
        :param lr: learning rate, defaults to 0.0002
        :type lr: float, optional
        :param beta_1: beta 1 for Adam optimizer, defaults to 0.5
        :type beta_1: float, optional
        """
        self.nc = nc
        self.ngf = ngf
        self.blocks = blocks
        self.lr = lr
        self.beta_1 = beta_1
        self.cycle_criterion = self.get_cycle_criterion()

    @abstractmethod
    def _create_model(self):
        pass

    def _set_optimizer(self):
        self.optimizer = self.get_optimizer()

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(self.beta_1, 0.999)
        )

    def get_cycle_criterion(self):
        return torch.nn.L1Loss()

    def get_loss(self, fake):
        return torch.mean((fake - 1) ** 2)

    def forward(self, x):
        return self.conv(x)
