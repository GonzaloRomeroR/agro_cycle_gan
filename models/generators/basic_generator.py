import torch.nn as nn

from .base_generator import BaseGenerator
from .custom_models import ResBlock

norm_layer = nn.InstanceNorm2d


class BasicGenerator(BaseGenerator):
    """
    Basic Generator class
    """

    def _create_model(self):
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, self.ngf, 7, 1, 0),
            nn.InstanceNorm2d(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, 2 * self.ngf, 3, 2, 1),
            nn.InstanceNorm2d(2 * self.ngf),
            nn.ReLU(True),
            nn.Conv2d(2 * self.ngf, 4 * self.ngf, 3, 2, 1),
            nn.InstanceNorm2d(4 * self.ngf),
            nn.ReLU(True),
        ]
        for i in range(int(self.blocks)):
            layers.append(ResBlock(4 * self.ngf))
        layers.extend(
            [
                nn.ConvTranspose2d(4 * self.ngf, 4 * 2 * self.ngf, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.InstanceNorm2d(2 * self.ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(2 * self.ngf, 4 * self.ngf, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.InstanceNorm2d(self.ngf),
                nn.ReLU(True),
                nn.ReflectionPad2d(3),
                nn.Conv2d(self.ngf, 3, 7, 1, 0),
                nn.Tanh(),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)
