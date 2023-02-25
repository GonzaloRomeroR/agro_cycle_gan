from abc import ABC, abstractmethod

import torch
from utils.sys_utils import get_device

from utils.image_utils import upload_images_numpy, upload_images
from utils.sys_utils import suppress_sklearn_errors, suppress_tf_warnings

suppress_tf_warnings()
suppress_sklearn_errors()

import os
from typing import Any, Dict, List, Tuple

import numpy as np
from generate import ImageTransformer
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from numpy.typing import NDArray
from scipy.linalg import sqrtm

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
        pass

    @abstractmethod
    def get_score(self, images_real: NDArray[Any], images_gen: NDArray[Any]) -> float:
        pass

    @abstractmethod
    def calculate_score(
        self, images_real: NDArray[Any], images_gen: NDArray[Any]
    ) -> float:
        pass

    def calculate_metrics(
        self, name: str, im_size: Tuple[int, ...], out_domain: str = "B"
    ) -> float:
        """Calculate metrics

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
        in_domain = "A" if out_domain == "B" else "B"
        # Generate images
        image_transformer = ImageTransformer(name)
        image_transformer.transform_dataset(
            f"./images/{name}/test_{in_domain}/{in_domain}/", f"./images_gen/{name}/"
        )

        # Uploading images
        fake_B = upload_images_numpy(f"./images_gen/{name}/", im_size=im_size)
        test_B = upload_images_numpy(
            f"./images/{name}/test_{out_domain}/{out_domain}/", im_size=im_size
        )
        # Get score
        score = self.get_score(test_B, fake_B)
        return score


class FID(Metrics):
    """
    Class to calculate the frechnet inception score
    """

    def _set_params(self, input_shape: Tuple[int, ...] = (299, 299, 3)) -> None:
        self.name = "FID"

        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        self.model = InceptionV3(
            include_top=False, pooling="avg", input_shape=input_shape
        )
        self.input_shape = input_shape

    def calculate_metrics(
        self, name: str, im_size: Tuple[int, ...], out_domain: str = "B"
    ) -> float:
        """Calculate metrics

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

        features_real = []
        features_gen = []

        in_domain = "A" if out_domain == "B" else "B"

        # Generate images
        image_transformer = ImageTransformer(name)

        test_images = upload_images(
            f"./images/{name}/test_{in_domain}/", im_size=im_size, batch_size=1
        )

        real_images = upload_images(
            f"./images/{name}/test_{out_domain}/", im_size=im_size
        )

        for image in test_images:
            image = image[0].to(get_device())
            images_gen = image_transformer.transform_image(image)
            images_gen = images_gen.detach().numpy().astype("float32")
            images_gen = self.scale_images(images_gen, self.input_shape)
            images_gen = preprocess_input(images_gen)
            features_gen.append(self.model.predict(images_gen))

        features_gen = np.array(features_gen)

        features_gen = features_gen.reshape(
            (features_gen.shape[0] * features_gen.shape[1], features_gen.shape[2])
        )

        for image in real_images:
            image = image[0].to(get_device())
            image = image.detach().numpy().astype("float32")
            image = self.scale_images(image, self.input_shape)
            image = preprocess_input(image)
            features_real.append(self.model.predict(image))

        features_real = np.array(features_real)

        features_real = features_real.reshape(
            (features_real.shape[0] * features_real.shape[1], features_real.shape[2])
        )

        return self.calculate_score(features_real, features_gen)

    def scale_images(
        self, images: NDArray[Any], new_shape: Tuple[int, ...]
    ) -> NDArray[Any]:
        images_list = list()
        for image in images:
            new_image = np.resize(image, new_shape)
            images_list.append(new_image)
        return np.asarray(images_list)

    def get_score(self, images_real: NDArray[Any], images_gen: NDArray[Any]):
        """
        Get frechnet inception distance

        :param model: model used to predict the feature vector
        :type model: Model
        :param images_real: collections of real images
        :type images_real: numpy array
        :param images_gen: collections of generated images
        :type images_gen: numpy array
        :return: frechnet inception distance
        :rtype: float
        """
        # Numpy array
        images_real = images_real.astype("float32")
        images_gen = images_gen.astype("float32")
        # Tensor
        # images_real = images_real.detach().numpy().astype("float32")
        # images_gen = images_gen.detach().numpy().astype("float32")
        images_real = self.scale_images(images_real, self.input_shape)
        images_gen = self.scale_images(images_gen, self.input_shape)
        images_real = preprocess_input(images_real)
        images_gen = preprocess_input(images_gen)
        # Calculate activations
        act1 = self.model.predict(images_real)
        act2 = self.model.predict(images_gen)
        return self.calculate_score(act1, act2)

    def calculate_score(
        self, features_real: NDArray[Any], features_gen: NDArray[Any]
    ) -> float:
        # Calculate mean and covariance statistics
        mu1, sigma1 = features_real.mean(axis=0), np.cov(features_real, rowvar=False)
        mu2, sigma2 = features_gen.mean(axis=0), np.cov(features_gen, rowvar=False)
        # Calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2)
        # Calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # Check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # Calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return float(np.abs(fid))
