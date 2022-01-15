import torch.nn as nn
import torch.nn.functional as F

from .base_generator import BaseGenerator


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


class BasicGenerator(BaseGenerator):
    """
    Basic Generator class
    """

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
