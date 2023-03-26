import time
import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
from models.discriminators.base_discriminator import BaseDiscriminator
from models.generators.base_generator import BaseGenerator
from utils.file_utils import save_models
from utils.metrics_utils import Metrics
from utils.plot_utils import plot_generator_images, plot_metrics
from utils.report_utils import ParamsLogger
from utils.sys_utils import get_gpu_usage
from utils.tensorboard_utils import TensorboardHandler
from utils.report_utils import generate_report
from utils.image_utils import DatasetDataloaders


class BaseTrainer(ABC):
    """
    Base Trainer class
    """

    def __init__(
        self,
        G_A2B: BaseGenerator,
        G_B2A: BaseGenerator,
        D_A: BaseDiscriminator,
        D_B: BaseDiscriminator,
        models_path: str,
        datasets: DatasetDataloaders,
        device: torch.device,
        params_logger: ParamsLogger,
        bs: int = 5,
        num_epochs: int = 20,
        plot_epochs: int = 1,
        print_info: int = 3,
        metrics: Optional[Metrics] = None,
        im_size: Tuple[int, ...] = (64, 64),
        tensorboard: bool = False,
        plot_image_epoch=False,
    ) -> None:
        """Train the generator and the discriminator

        :param G_A2B: generator to transform images from A to B
        :type G_A2B: `Generator`
        :param G_B2A: generator to transform images from B to A
        :type G_B2A: `Generator`
        :param D_A: discriminator for images in the domain A
        :type D_A: `Discriminator`
        :param D_B: discriminator for images in the domain A
        :type D_B: `Discriminator`
        :param datasets: object with dataloaders with images
        :type datasets: ´DatasetDataloaders´
        :param models_path: path to load and save the models
        :type models_path: str
        :param device: pytorch device
        :type device: ´Device´
        :param bs: batch size, defaults to 5
        :type bs: int, optional
        :param num_epochs: number of epochs, defaults to 20
        :type num_epochs: int, optional
        :param plot_epochs: epochs before printing images, defaults to 5
        :type plot_epochs: int, optional
        :param print_info: batchs processed before printing losses, defaults to 3
        :type print_info: int, optional
        :param params_logger: logger to store the training information, defaults to None
        :type params_logger: `ParamsLogger`, optional
        :param metrics: logger to store the training information, defaults to None
        :type metrics: `Metrics`, optional
        :param tensorboad: flag to store in tensorboard, defaults to False
        :type metrics: bool, optional
        :param plot_image_epoch: flag to plot image after epoch, defaults to False
        :type metrics: bool, optional
        """
        self.G_A2B = G_A2B
        self.G_B2A = G_B2A
        self.D_A = D_A
        self.D_B = D_B
        self.models_path = models_path
        self.images_A = datasets.train_A
        self.images_B = datasets.train_B
        self.dataset_name = datasets.name
        self.device = device
        self.test_images_A = datasets.test_A
        self.test_images_B = datasets.test_B
        self.bs = bs
        self.num_epochs = num_epochs
        self.plot_epochs = plot_epochs
        self.print_info = print_info
        self.params_logger = params_logger
        self.metrics = metrics
        self.im_size = im_size
        self._set_training_params()
        self._define_storing()
        self.tensorboard = tensorboard
        self.plot_image_epoch = plot_image_epoch
        self.metrics_per_epoch = {}
        if self.tensorboard:
            self._set_tensorboard()

    @abstractmethod
    def _set_training_params(self) -> None:
        """
        Set parameters for training
        """

    def _define_storing(self) -> None:
        self.D_A_losses: List[Any] = []
        self.D_B_losses: List[Any] = []
        self.G_losses: List[Any] = []

        self.losses_names = [
            "FDL_A2B",
            "FDL_B2A",
            "CL_A",
            "CL_B",
            "ID_B2A",
            "ID_A2B",
            "GEN_TOTAL",
            "DISC_A",
            "DISC_B",
        ]

        self.losses_epoch: Dict[str, Any] = {key: [] for key in self.losses_names}
        self.losses_total: Dict[str, Any] = {key: [] for key in self.losses_names}

    def _set_tensorboard(self) -> None:
        self.writer = TensorboardHandler("./runs/Losses")

    def generate_images_cycle(
        self,
        a_real: torch.Tensor,
        b_real: torch.Tensor,
        G_A2B: BaseGenerator,
        G_B2A: BaseGenerator,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Create batches of fake and reconstructed images
        based on batches of real images as input.
        """
        b_fake = G_A2B(a_real)
        a_recon = G_B2A(b_fake)
        a_fake = G_B2A(b_real)
        b_recon = G_A2B(a_fake)
        return a_fake, b_fake, a_recon, b_recon

    def _print_iter_info(
        self,
        epoch: int,
        iteration: int,
        gen_losses: Dict[str, Any],
        disc_losses: Dict[str, Any],
    ) -> None:
        info = f"Epoch [{epoch}/{self.num_epochs}] batch [{iteration}]"
        for name, loss in {**gen_losses, **disc_losses}.items():
            info += f" {name}: {loss:.3f}"
        print(info)

    def convert_total_time_format(self, seconds: float) -> str:
        """
        Converts seconds to hour minute seconds format
        """
        hours = seconds // 3600
        seconds -= hours * 3600
        minutes = seconds // 60
        seconds -= minutes * 60
        return f"{hours} h {minutes} m {seconds} s"

    def _save_epoch_information(self, epoch: int) -> None:
        # Generate epoch information
        self.params_logger.params["final_epoch"] = epoch
        total_time = round(time.perf_counter() - self.start_time)
        self.params_logger.params["time"] = str(
            self.convert_total_time_format(total_time)
        )
        self.params_logger.generate_params_file()

    def _obtain_total_losses(self, epoch: int) -> None:
        # Obtain total losses
        for key in self.losses_epoch.keys():
            loss_key = sum(self.losses_epoch[key]) / len(self.losses_epoch[key])
            self.losses_total[key].append(loss_key)
            if self.tensorboard:
                self.writer.add_scalar(key, loss_key, epoch)

        # Clear epoch list
        self.losses_epoch = {key: [] for key in self.losses_names}

    def _run_post_epoch(self, epoch: int) -> None:

        self._obtain_total_losses(epoch)
        self._save_epoch_information(epoch)

        save_models(
            self.models_path,
            self.dataset_name,
            self.G_A2B,
            self.G_B2A,
            self.D_A,
            self.D_B,
        )
        if epoch % self.plot_epochs == 0 and self.plot_image_epoch:
            plot_generator_images(
                self.G_A2B,
                self.G_B2A,
                self.test_images_A,
                self.test_images_B,
                self.device,
            )

        # Get gpu usage
        print(get_gpu_usage())

        # Obtain metrics
        if self.metrics:
            score = self.metrics.calculate_metrics(self.dataset_name, self.im_size[1:])
            print(f"{self.metrics.name} score: {score}")

            if self.metrics.name not in self.metrics_per_epoch:
                self.metrics_per_epoch[self.metrics.name] = []

            self.metrics_per_epoch[self.metrics.name].append(score)

            if epoch % self.plot_epochs == 0 and self.plot_image_epoch:
                plot_metrics(self.metrics_per_epoch)

        if epoch != 0 and epoch % 10 == 0:
            generate_report(
                self.params_logger.params, self.losses_total, self.metrics_per_epoch
            )

    def train(self) -> Dict[str, Any]:
        """
        Train model getting the training losses
        """
        self.start_time = time.perf_counter()
        self._train_model()
        end_time = self.convert_total_time_format(
            round(time.perf_counter() - self.start_time)
        )
        print(f"Full trainig took {end_time} to finish")
        return self.losses_total

    @abstractmethod
    def _train_model(self) -> None:
        pass
