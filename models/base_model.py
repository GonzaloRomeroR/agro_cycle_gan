from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Base discriminator class
    """

    def __init__(self):
        super().__init__()
        self._set_params()
        self._create_model()

    @abstractmethod
    def _set_params(self):
        return

    @abstractmethod
    def _create_model(self):
        return

    @abstractmethod
    def forward():
        return

