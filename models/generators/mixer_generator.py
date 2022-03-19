import torch.nn as nn

from .base_generator import BaseGenerator
from .custom_models import MixerBlock, View


class MixerGenerator(BaseGenerator):
    """
    Mixer Generator class
    """

    def _set_custom_params(
        self,
        patch_dim=64,
        image_size=256,
        embed_dim=256,
        transform_layers=9,
        patch_size=8,
    ) -> None:
        self.patch_dim = patch_dim
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.transform_layers = transform_layers
        self.patch_size = patch_size

    def _create_model(self) -> None:
        # Stem
        model = [
            nn.Conv2d(
                in_channels=self.n_channels,
                out_channels=self.embed_dim // 4,
                kernel_size=7,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(self.embed_dim // 4),
            nn.ReLU(True),
        ]
        # Downsampling
        model += [
            nn.Conv2d(
                in_channels=self.embed_dim // 4,
                out_channels=self.embed_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(self.embed_dim // 2),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=self.embed_dim // 2,
                out_channels=self.embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(self.embed_dim),
            nn.ReLU(True),
        ]
        # Linear down-projection
        model += [
            nn.Conv2d(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
        ]
        # Reshape to (tokens, channels)
        model += [View((self.embed_dim, self.patch_dim))]
        # Transformation
        model += [
            MixerBlock(embed_dim=self.embed_dim, patch_dim=self.patch_dim)
            for _ in range(self.transform_layers)
        ]
        # Reshape to (c, h, w)
        model += [
            View(
                (
                    self.embed_dim,
                    self.image_size // 4 // self.patch_size,
                    self.image_size // 4 // self.patch_size,
                )
            )
        ]
        # Linear up-projection
        model += [
            nn.ConvTranspose2d(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
        ]
        # Upsampling
        model += [
            nn.ConvTranspose2d(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.InstanceNorm2d(self.embed_dim // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=self.embed_dim // 2,
                out_channels=self.embed_dim // 4,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.InstanceNorm2d(self.embed_dim // 4),
            nn.ReLU(True),
        ]
        # To RGB
        model += [
            nn.Conv2d(
                in_channels=self.embed_dim // 4,
                out_channels=3,
                kernel_size=7,
                padding=3,
                padding_mode="reflect",
            )
        ]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
