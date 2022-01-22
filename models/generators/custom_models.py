import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Resblock class
    """

    def __init__(self, nf):
        self.nf = nf
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(self.nf, self.nf, 3, 1, 1),
            nn.InstanceNorm2d(self.nf),
            nn.ReLU(),
            nn.Conv2d(self.nf, self.nf, 3, 1, 1),
        )
        self.norm = nn.InstanceNorm2d(nf)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x) + x))
