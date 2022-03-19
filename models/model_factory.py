from .discriminators.basic_discriminator import BasicDiscriminator
from .generators.cyclegan_generator import CycleganGenerator
from .generators.mixer_generator import MixerGenerator


class ModelCreator:
    """
    Model factory
    """

    def create(self, model_type: str, model_name: str, **kwargs):
        """Create and return model

        :param model_type: model type, gen or disc
        :type model_type: str
        :param model_name: model name to use
        :type model_name: str
        """
        if model_type == "gen":
            if model_name == "mixer":
                return MixerGenerator(**kwargs)
            else:
                return CycleganGenerator(**kwargs)
        elif model_type == "disc":
            if model_name == "example":
                return BasicDiscriminator(**kwargs)
            else:
                return BasicDiscriminator(**kwargs)
        else:
            raise Exception("Model type not supported")

