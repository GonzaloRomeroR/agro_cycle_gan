from abc import ABC, abstractmethod

import torch
from utils.sys_utils import get_device

from utils.image_utils import upload_images_numpy, upload_images
from utils.sys_utils import suppress_sklearn_errors, suppress_tf_warnings

from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image

suppress_tf_warnings()
suppress_sklearn_errors()

import os
from typing import Any, Dict, List, Tuple

from generate import ImageTransformer
from numpy.typing import NDArray

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Metrics(ABC):
    """
    Base class for calculating metrics
    """

    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        self._set_params(*args, **kwargs)
        self.name: str

    @abstractmethod
    def _set_params(self, *args: List[Any], **kwargs: Dict[str, Any]) -> None:
        """
        Set parameter for metrics calculations
        """
        pass

    @abstractmethod
    def _get_score(self, images_real: NDArray[Any], images_gen: NDArray[Any]) -> float:
        """
        Get score based on numpy arrays of the real and generated images
        :param images_real: collections of real images
        :type images_real: numpy array
        :param images_gen: collections of generated images
        :type images_gen: numpy array
        :return: metric score
        :rtype: float
        """
        pass

    @abstractmethod
    def calculate_metrics(
        self, name: str, im_size: Tuple[int, ...], out_domain: str = "B"
    ) -> float:
        """Calculate metrics using the training test folder

        :param metrics: metrics to compute
        :type metrics: ``Metrics`
        :param name: name of the dataset to compute
        :type name: str
        :param im_size: size of the images
        :type im_size: tuple (h, w)
        :param out_domain: target domain to produce the transformation, defaults to "B"
        :type out_domain: str, optional
        :return: obtain metrics
        :rtype: float
        """
        pass


class FID_Pytorch_TorchMetrics(Metrics):
    """
    Class to calculate the frechnet inception score
    """

    def _set_params(self, input_shape: Tuple[int, ...] = (3, 299, 299)) -> None:
        self.name = "FID"
        self.fid = FrechetInceptionDistance(feature=64)
        self.input_shape = input_shape

    def _get_score(self, images_real: NDArray[Any], images_gen: NDArray[Any]):
        """
        Get frechnet inception score
        """
        self.fid.update(images_real, real=True)
        self.fid.update(images_gen, real=False)
        return self.fid.compute()

    def calculate_metrics(
        self, name: str, im_size: Tuple[int, ...], out_domain: str = "B"
    ) -> float:

        in_domain = "A" if out_domain == "B" else "B"
        # Generate images
        image_transformer = ImageTransformer(name)

        test_images = upload_images(
            f"./images/{name}/test_{in_domain}/",
            im_size=im_size,
            batch_size=1,
            data_augmentation=False,
        )

        for i, image in enumerate(test_images):
            print(f"Generated image {i}")
            image = image[0].to(get_device())
            images_gen = image_transformer.transform_image(image)
            save_image(images_gen, f"./images_gen/{name}/{i}.jpg")

        # Uploading images
        fake_B = upload_images_numpy(f"./images_gen/{name}/", im_size=im_size)
        test_B = upload_images_numpy(
            f"./images/{name}/test_{out_domain}/{out_domain}/", im_size=im_size
        )

        fake_B = torch.from_numpy(fake_B).permute(0, 3, 1, 2)
        test_B = torch.from_numpy(test_B).permute(0, 3, 1, 2)
        # Get score
        score = self._get_score(test_B, fake_B)
        return score


def create_metrics(name: str) -> Metrics:
    if name == "FID_TensorFlow_DataLoader":
        return FID_Pytorch_TorchMetrics()
    else:
        return FID_Pytorch_TorchMetrics()  # Right now only this metric is available
