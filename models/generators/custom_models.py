from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Resblock class
    """

    def __init__(self, nf: int):
        self.nf = nf
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.nf,
                out_channels=self.nf,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm2d(num_features=self.nf),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.nf,
                out_channels=self.nf,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.norm = nn.InstanceNorm2d(nf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.conv(x) + x))


class View(nn.Module):
    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size = input.size(0)
        tensor_shape: Tuple[int, ...] = (batch_size, *self.shape)
        out = input.view(tensor_shape)
        return out


class MixerBlock(nn.Module):
    def __init__(self, embed_dim: int, patch_dim: int):
        super(MixerBlock, self).__init__()
        self.ln1 = nn.LayerNorm([embed_dim, patch_dim])
        self.dense1 = nn.Linear(
            in_features=patch_dim, out_features=patch_dim, bias=False
        )
        self.gelu1 = nn.GELU()
        self.dense2 = nn.Linear(
            in_features=patch_dim, out_features=patch_dim, bias=False
        )
        self.ln2 = nn.LayerNorm([embed_dim, patch_dim])
        # Using conv1d with kernel_size=1 is like applying a
        # linear layer to the channel dim
        self.conv1 = nn.Conv1d(
            in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=False
        )
        self.gelu2 = nn.GELU()
        self.conv2 = nn.Conv1d(
            in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token-mixing mlp
        skip = x
        x = self.ln1(x)
        x = self.dense1(x)
        x = self.gelu1(x)
        x = self.dense2(x)
        x = x + skip

        # Channel-mixing mlp
        skip = x
        x = self.ln2(x)
        x = self.conv1(x)
        x = self.gelu2(x)
        x = self.conv2(x)
        x = x + skip
        return x
