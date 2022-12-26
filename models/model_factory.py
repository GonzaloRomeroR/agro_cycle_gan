from typing import Any, Dict, Union

from .discriminators.base_discriminator import BaseDiscriminator
from .discriminators.basic_discriminator import BasicDiscriminator
from .generators.base_generator import BaseGenerator
from .generators.cyclegan_generator import CycleganGenerator
from .generators.mixer_generator import MixerGenerator


class ModelCreator:
    """
    Model factory
    """

    def create(
        self, model_type: str, model_name: str, **kwargs: Dict[str, Any]
    ) -> Union[BaseDiscriminator, BaseGenerator]:
        """Create and return model

        :param model_type: model type, gen or disc
        :type model_type: str
        :param model_name: model name to use
        :type model_name: str
        """
        if model_type == "gen":
            if model_name == "mixer":
                return MixerGenerator(**kwargs)
            elif model_name == "cyclegan":
                return CycleganGenerator(**kwargs)
            else:
                return CycleganGenerator(**kwargs)
        elif model_type == "disc":
            if model_name == "basic":
                return BasicDiscriminator(**kwargs)
            else:
                return BasicDiscriminator(**kwargs)
        else:
            raise Exception("Model type not supported")
