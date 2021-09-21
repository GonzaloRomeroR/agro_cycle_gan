from .generator import Generator
from .discriminator import Discriminator


class ModelCreator:
    """
    Model factory
    """

    def create(self, model_type, model_name, **kwargs):
        """Create and return model

        :param model_type: model type, gen or disc
        :type model_type: str
        :param model_name: model name to use
        :type model_name: str
        """
        if model_type == "gen":
            if model_name == "example":
                return Generator(**kwargs)
            else:
                return Generator(**kwargs)
        elif model_type == "disc":
            if model_name == "example":
                return Discriminator(**kwargs)
            else:
                return Discriminator(**kwargs)
        else:
            raise Exception("Model type not supported")

