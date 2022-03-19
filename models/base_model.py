from abc import ABC, abstractmethod
from typing import Any, Dict

import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Base discriminator class
    """

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self._set_params(**kwargs)
        self._create_model()
        self._set_optimizer()

    @abstractmethod
    def _set_params(self, **kwargs: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def _set_optimizer(self) -> None:
        pass

    @abstractmethod
    def _create_model(self) -> None:
        pass

    @abstractmethod
    def get_loss(self) -> None:
        pass

    @abstractmethod
    def forward(self) -> None:
        pass

