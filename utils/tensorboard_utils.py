from torch.utils.tensorboard import SummaryWriter


class TensorboardHandler:
    """
    Class to handle tensorboard functions
    """

    _instance = {}

    # Singleton to create only one writer per model
    def __new__(class_, name=None, *args, **kwargs):
        if name not in class_._instance.keys():
            class_._instance[name] = object.__new__(class_, *args, **kwargs)
        return class_._instance[name]

    def __init__(self, name=None):
        self.name = name
        self.writer = SummaryWriter(name)

    def add_graph(self, model, images):
        self.writer.add_graph(model, images)

    def add_image(self, grid):
        self.writer.add_image("images", grid, 0)

    def add_scalar(self, dir_name, value, n_iter):
        self.writer.add_scalar(f"./runs/{self.name}/{dir_name}", value, n_iter)


def create_models_tb(G_A2B, G_B2A, D_A, D_B, images):
    model_dict = {"G_A2B": G_A2B, "G_B2A": G_B2A, "D_A": D_A, "D_B": D_B}
    for model_name in model_dict.keys():
        tb_model = TensorboardHandler(f"./runs/{model_name}")
        tb_model.add_graph(model_dict[model_name], images)

