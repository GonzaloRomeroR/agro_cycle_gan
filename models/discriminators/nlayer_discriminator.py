import torch.nn as nn
import functools

from .base_discriminator import BaseDiscriminator


class NLayerDiscriminator(BaseDiscriminator):
    """
    NLayer Discriminator class
    """

    def _set_custom_params(self, n_layers: int = 3, norm_layer=nn.BatchNorm2d) -> None:
        self.n_layers = n_layers
        self.norm_layer = norm_layer

    def _create_model(self) -> None:
        if (
            type(self.norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = self.norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = self.norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(self.nc, self.ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, self.n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    self.ndf * nf_mult_prev,
                    self.ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                self.norm_layer(self.ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**self.n_layers, 8)
        sequence += [
            nn.Conv2d(
                self.ndf * nf_mult_prev,
                self.ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            self.norm_layer(self.ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(self.ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)
