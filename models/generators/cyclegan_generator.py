import torch.nn as nn

from .base_generator import BaseGenerator
from .custom_models import ResBlock


class CycleganGenerator(BaseGenerator):
    """
    Basic Generator class
    """

    def _set_custom_params(self, filters=64, blocks=9):
        self.filters = filters
        self.blocks = blocks

    def _create_model(self):
        layers = [
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(
                in_channels=self.n_channels,
                out_channels=self.filters,
                kernel_size=7,
                stride=1,
                padding=0,
            ),
            nn.InstanceNorm2d(num_features=self.filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.filters,
                out_channels=2 * self.filters,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(num_features=2 * self.filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=2 * self.filters,
                out_channels=4 * self.filters,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(num_features=4 * self.filters),
            nn.ReLU(inplace=True),
        ]
        for _ in range(int(self.blocks)):
            layers.append(ResBlock(4 * self.filters))
        layers.extend(
            [
                nn.ConvTranspose2d(
                    in_channels=4 * self.filters,
                    out_channels=4 * 2 * self.filters,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.PixelShuffle(upscale_factor=2),
                nn.InstanceNorm2d(num_features=2 * self.filters),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=2 * self.filters,
                    out_channels=4 * self.filters,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.PixelShuffle(upscale_factor=2),
                nn.InstanceNorm2d(num_features=self.filters),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(padding=3),
                nn.Conv2d(
                    in_channels=self.filters,
                    out_channels=3,
                    kernel_size=7,
                    stride=1,
                    padding=0,
                ),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
