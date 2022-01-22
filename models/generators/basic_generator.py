import torch.nn as nn

from .base_generator import BaseGenerator
from .custom_models import ResBlock


class BasicGenerator(BaseGenerator):
    """
    Basic Generator class
    """
    def _set_custom_params(self, filters=64, blocks=9):
        self.filters = filters
        self.blocks = blocks

    def _create_model(self):
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.n_channels, self.filters, 7, 1, 0),
            nn.InstanceNorm2d(self.filters),
            nn.ReLU(True),
            nn.Conv2d(self.filters, 2 * self.filters, 3, 2, 1),
            nn.InstanceNorm2d(2 * self.filters),
            nn.ReLU(True),
            nn.Conv2d(2 * self.filters, 4 * self.filters, 3, 2, 1),
            nn.InstanceNorm2d(4 * self.filters),
            nn.ReLU(True),
        ]
        for i in range(int(self.blocks)):
            layers.append(ResBlock(4 * self.filters))
        layers.extend(
            [
                nn.ConvTranspose2d(4 * self.filters, 4 * 2 * self.filters, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.InstanceNorm2d(2 * self.filters),
                nn.ReLU(True),
                nn.ConvTranspose2d(2 * self.filters, 4 * self.filters, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.InstanceNorm2d(self.filters),
                nn.ReLU(True),
                nn.ReflectionPad2d(3),
                nn.Conv2d(self.filters, 3, 7, 1, 0),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
