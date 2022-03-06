import torch.nn as nn

from .base_discriminator import BaseDiscriminator


class BasicDiscriminator(BaseDiscriminator):
    """
    Basic Discriminator class
    """

    def _create_model(self) -> None:
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels=self.nc,
                out_channels=self.ndf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=self.ndf,
                out_channels=self.ndf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(num_features=self.ndf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=self.ndf * 2,
                out_channels=self.ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(num_features=self.ndf * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=self.ndf * 4,
                out_channels=self.ndf * 8,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm2d(num_features=self.ndf * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=self.ndf * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
        )
