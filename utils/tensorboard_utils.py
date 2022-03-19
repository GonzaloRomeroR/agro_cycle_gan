from typing import Any, Dict, List, Optional

from torch.utils.tensorboard import SummaryWriter


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

    def add_graph(self, model, images) -> None:
        self.writer.add_graph(model, images)

    def add_image(self, grid: Any) -> None:
        self.writer.add_image("images", grid, 0)

    def add_scalar(self, dir_name: str, value: float, n_iter: int) -> None:
        self.writer.add_scalar(f"./runs/{self.name}/{dir_name}", value, n_iter)


def create_models_tb(G_A2B, G_B2A, D_A, D_B, images) -> None:
    """Create tensorboard runs to store models

    :param G_A2B: Generator from A to B
    :type G_A2B: `Generator`
    :param G_B2A: Generator from B to A
    :type G_B2A: `Generator`
    :param D_A: Discriminator for A
    :type D_A: `Discriminator`
    :param D_B: Discriminator for B
    :type D_B: `Discriminator`
    :param images: Image to get the size
    :type images: Tensor
    """
    model_dict = {"G_A2B": G_A2B, "G_B2A": G_B2A, "D_A": D_A, "D_B": D_B}
    for model_name in model_dict.keys():
        tb_model = TensorboardHandler(f"./runs/{model_name}")
        tb_model.add_graph(model_dict[model_name], images)

