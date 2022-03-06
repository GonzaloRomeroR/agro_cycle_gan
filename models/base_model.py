from abc import ABC, abstractmethod

import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Base discriminator class
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._set_params(**kwargs)
        self._create_model()
        self._set_optimizer()

    @abstractmethod
    def _set_params(self, **kwargs):
        pass

    @abstractmethod
    def _set_optimizer(self):
        pass

    @abstractmethod
    def _create_model(self):
        pass

    @abstractmethod
    def get_loss(self):
        pass

    @abstractmethod
    def forward(self):
        pass

