from abc import ABC, abstractmethod
from utils.sys_utils import suppress_tf_warnings, surpress_sklearn_errors

suppress_tf_warnings()
surpress_sklearn_errors()

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import numpy as np
from scipy.linalg import sqrtm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Metrics(ABC):
    """
    Base class for calculating metrics
    """

    def __init__(self, *args, **kwargs):
        self._set_params(*args, **kwargs)

    @abstractmethod
    def _set_params(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_score(self, images_real, images_gen):
        pass

    @abstractmethod
    def calculate_score(self, images_real, images_gen):
        pass


class FID(Metrics):
    """
    Class to calculate the frechnet inception score 
    """

    def _set_params(self, input_shape=(299, 299, 3)):
        self.model = InceptionV3(
            include_top=False, pooling="avg", input_shape=input_shape
        )
        self.input_shape = input_shape

    def scale_images(self, images, new_shape):
        images_list = list()
        for image in images:
            new_image = np.resize(image, new_shape)
            images_list.append(new_image)
        return np.asarray(images_list)

    def get_score(self, images_real, images_gen):
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
        images_real = images_real.astype("float32")
        images_gen = images_gen.astype("float32")
        images_real = self.scale_images(images_real, self.input_shape)
        images_gen = self.scale_images(images_gen, self.input_shape)
        images_real = preprocess_input(images_real)
        images_gen = preprocess_input(images_gen)
        fid = self.calculate_score(images_real, images_gen)
        return fid

    def calculate_score(self, images_real, images_gen):
        # Calculate activations
        act1 = self.model.predict(images_real)
        act2 = self.model.predict(images_gen)
        # Calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        # Calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2)
        # Calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # Check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # Calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return np.abs(fid)

