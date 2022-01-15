import torch.nn as nn

from .base_discriminator import BaseDiscriminator


class BasicDiscriminator(BaseDiscriminator):
    """
    BasicDiscriminator class
    """

    def _create_model(self):
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf) x 64 x 64
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*2) x 32 x 32
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*4) x 16 x 16
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 1, 1),
            nn.InstanceNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*8) x 15 x 15
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 1)
            # state size. 1 x 14 x 14
        )
