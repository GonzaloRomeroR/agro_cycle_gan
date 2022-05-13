from abc import abstractmethod
from typing import Any, Dict

import torch

from ..base_model import BaseModel


class BaseGenerator(BaseModel):
    """
    Base Generator class
    """

    def _set_params(
        self,
        n_channels: int = 3,
        lr: float = 0.0002,
        beta_1: float = 0.5,
        **kwargs: Dict[str, Any]
    ) -> None:
        """Set generator parameters

        :param ngf: size of feature maps in generator, defaults to 64
        :type ngf: int, optional
        :param n_channels: number of channels in the training images, defaults to 3
        :type n_channels: int, optional   
        :param blocks: number of ResBlocks, defaults to 9
        :type blocks: int, optional
        :param lr: learning rate, defaults to 0.0002
        :type lr: float, optional
        :param beta_1: beta 1 for Adam optimizer, defaults to 0.5
        :type beta_1: float, optional
        """
        self.n_channels = n_channels
        self.lr = lr
        self.beta_1 = beta_1
        self.cycle_criterion = self.get_cycle_criterion()
        self._set_custom_params(**kwargs)

    def _set_custom_params(self, **kwargs) -> None:
        pass

    def _set_optimizer(self) -> None:
        self.optimizer = self.get_optimizer()

    def get_optimizer(self) -> Any:
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(self.beta_1, 0.999)
        )

    def get_cycle_criterion(self) -> Any:
        return torch.nn.L1Loss()

    def get_loss(self, fake: torch.Tensor) -> torch.Tensor:
        return torch.mean((fake - 1) ** 2)

    @abstractmethod
    def forward(self, x: Any) -> Any:
        pass

    @abstractmethod
    def _create_model(self) -> None:
        pass
