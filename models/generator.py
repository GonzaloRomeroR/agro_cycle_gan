import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel


norm_layer = nn.InstanceNorm2d


class ResBlock(nn.Module):
    """
    Resblock class
    """

    def __init__(self, nf):
        self.nf = nf
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(self.nf, self.nf, 3, 1, 1),
            norm_layer(self.nf),
            nn.ReLU(),
            nn.Conv2d(self.nf, self.nf, 3, 1, 1),
        )
        self.norm = norm_layer(nf)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x) + x))


class Generator(BaseModel):
    """
    Generator class
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

    def _create_model(self):
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, self.ngf, 7, 1, 0),
            norm_layer(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, 2 * self.ngf, 3, 2, 1),
            norm_layer(2 * self.ngf),
            nn.ReLU(True),
            nn.Conv2d(2 * self.ngf, 4 * self.ngf, 3, 2, 1),
            norm_layer(4 * self.ngf),
            nn.ReLU(True),
        ]
        for i in range(int(self.blocks)):
            layers.append(ResBlock(4 * self.ngf))
        layers.extend(
            [
                nn.ConvTranspose2d(4 * self.ngf, 4 * 2 * self.ngf, 3, 1, 1),
                nn.PixelShuffle(2),
                norm_layer(2 * self.ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(2 * self.ngf, 4 * self.ngf, 3, 1, 1),
                nn.PixelShuffle(2),
                norm_layer(self.ngf),
                nn.ReLU(True),
                nn.ReflectionPad2d(3),
                nn.Conv2d(self.ngf, 3, 7, 1, 0),
                nn.Tanh(),
            ]
        )
        self.conv = nn.Sequential(*layers)

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
