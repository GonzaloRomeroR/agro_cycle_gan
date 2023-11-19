from typing import Any, Dict, List, Optional, Union

import torch
from models.base_model import BaseModel
from models.discriminators.base_discriminator import BaseDiscriminator
from models.generators.base_generator import BaseGenerator
from torch.utils.tensorboard import SummaryWriter
from .train_utils import Models


class TensorboardHandler:
    """
    Class to handle tensorboard functions
    """

    _instance: Dict[str, Any] = {}

    # Singleton to create only one writer per model
    def __new__(
        class_, name: Optional[str] = None, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> Any:
        if name not in class_._instance.keys():
            class_._instance[name] = object.__new__(class_, *args, **kwargs)
        return class_._instance[name]

    def __init__(self, name: Optional[str] = None) -> None:
        """Class constructor

        :param name: name of the run folder, defaults to None
        :type name: str, optional
        """
        self.name = name
        self.writer = SummaryWriter(name)

    def add_graph(self, model: BaseModel, images: torch.Tensor) -> None:
        self.writer.add_graph(model, images)

    def add_image(self, grid: Any) -> None:
        self.writer.add_image("images", grid, 0)

    def add_scalar(self, dir_name: str, value: float, n_iter: int) -> None:
        self.writer.add_scalar(f"./runs/{self.name}/{dir_name}", value, n_iter)


def create_models_tb(
    models: Models,
    images: torch.Tensor,
) -> None:
    """Create tensorboard runs to store models

    :param models: used models
    :type models: `Models`
    :param images: Image to get the size
    :type images: Tensor
    """
    model_dict = {"G_A2B": models.G_A2B, "G_B2A": models.G_B2A, "D_A": models.D_A, "D_B": models.D_B}
    for model_name in model_dict.keys():
        tb_model = TensorboardHandler(f"./runs/{model_name}")
        tb_model.add_graph(model_dict[model_name], images)

