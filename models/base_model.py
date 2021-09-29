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
        self._set_optimizer()

    @abstractmethod
    def _set_params(self):
        return

    @abstractmethod
    def _set_optimizer(self):
        return

    @abstractmethod
    def _create_model(self):
        return

    @abstractmethod
    def get_loss(self):
        return

    @abstractmethod
    def forward():
        return

