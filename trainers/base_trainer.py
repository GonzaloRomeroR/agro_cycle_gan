import time
import shutil
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from time import gmtime, strftime

import torch
from models.generators.base_generator import BaseGenerator
from utils.file_utils import save_models
from utils.plot_utils import plot_generator_images, plot_metrics
from utils.report_utils import ParamsLogger
from utils.sys_utils import get_gpu_usage
from utils.tensorboard_utils import TensorboardHandler
from utils.report_utils import ResultsReporter
from utils.image_utils import DatasetDataloaders
from utils.train_utils import Config, Models
from utils.metrics_utils import create_metrics


class BaseTrainer(ABC):
    """
    Base Trainer class
    """

    def __init__(
        self,
        models: Models,
        models_path: str,
        datasets: DatasetDataloaders,
        config: Config,
        device: torch.device,
        params_logger: ParamsLogger,
        plot_epochs: int = 1,
        print_info: int = 3,
        im_size: Tuple[int, ...] = (64, 64),
    ) -> None:
        """Train the generator and the discriminator

        :param models: models to use
        :type models: `Models`
        :param datasets: object with dataloaders with images
        :type datasets: ´DatasetDataloaders´
        :param models_path: path to load and save the models
        :type models_path: str
        :param device: pytorch device
        :type device: ´Device´
        :param plot_epochs: epochs before printing images, defaults to 5
        :type plot_epochs: int, optional
        :param print_info: batchs processed before printing losses, defaults to 3
        :type print_info: int, optional
        :param params_logger: logger to store the training information, defaults to None
        :type params_logger: `ParamsLogger`, optional
        """
        self.models_path = models_path
        self.device = device
        self.plot_epochs = plot_epochs
        self.print_info = print_info
        self.params_logger = params_logger
        self.im_size = im_size
        self._set_models(models)
        self._set_datasets_config(datasets)
        self._set_config(config)
        self._set_training_params()
        self._define_storing()
        self.metrics_per_epoch = {}
    
    def _set_datasets_config(self, datasets: DatasetDataloaders):
        """
        Set datasets configuration
        """
        self.dataset_name = datasets.name
        self.images_A = datasets.train_A
        self.images_B = datasets.train_B
        self.test_images_A = datasets.test_A
        self.test_images_B = datasets.test_B
        

    def _set_config(self, config: Config):
        """
        Set trainer configuration
        """
        self.bs = config.batch_size
        self.num_epochs = config.num_epochs
        self.plot_image_epoch = config.plot_image_epoch
        self.store_models = config.store_models
        self.tensorboard = config.tensorboard
        self.metrics = create_metrics(config.metrics)
        if self.tensorboard:
            self._set_tensorboard()
    
    def _set_models(self, models: Models):
        """
        Set models configuration
        """
        self.G_A2B = models.G_A2B
        self.G_B2A = models.G_B2A
        self.D_A = models.D_A
        self.D_B = models.D_B


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
        self.losses_writer = TensorboardHandler("./runs/Losses")
        self.metrics_writer = TensorboardHandler("./runs/Metrics")

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
        """
        Print information about the current iteration
        """
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
        """
        Store information about the current epoch and regenerate parameter file
        """
        self.params_logger.params["final_epoch"] = epoch
        total_time = round(time.perf_counter() - self.start_time)
        self.params_logger.params["time"] = str(
            self.convert_total_time_format(total_time)
        )
        self.params_logger.generate_params_file()

    def _obtain_total_losses(self, epoch: int) -> None:
        """
        Obtain total losses of the epoch based on all the losses of the training
        """
        for key in self.losses_epoch.keys():
            loss = sum(self.losses_epoch[key]) / len(self.losses_epoch[key])
            self.losses_total[key].append(loss)
            if self.tensorboard:
                self.losses_writer.add_scalar(key, loss, epoch)

        # Clear epoch list
        self.losses_epoch = {key: [] for key in self.losses_names}

    def _run_post_epoch(self, epoch: int) -> None:

        """
        Run tasks needed after eposh is finished
        """
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

        if self.store_models and epoch % self.store_models == 0:
            shutil.copytree(
                f"./results/{self.dataset_name}",
                f"./results/{self.dataset_name}/models_{str(strftime('%Y-%m-%d--%H-%M-%S', gmtime()))}",
            )

        # Get gpu usage
        print(get_gpu_usage())

        # Obtain metrics
        if self.metrics:
            self._obtain_metrics(epoch)

        if epoch != 0 and epoch % 10 == 0:
            ResultsReporter.generate_report(
                self.dataset_name,
                self.params_logger.params,
                self.losses_total,
                self.metrics_per_epoch,
            )

    def _calculate_metrics_for_epoch(self, name: str, epoch: int, out_domain: str):
        score = self.metrics.calculate_metrics(
            self.dataset_name, self.im_size[1:], out_domain
        )
        print(f"{name} score: {score}")

        if name not in self.metrics_per_epoch:
            self.metrics_per_epoch[name] = []

        self.metrics_per_epoch[name].append(score)

        if self.tensorboard:
            self.metrics_writer.add_scalar(name, score, epoch)

    def _obtain_metrics(self, epoch: int):
        """
        Obtain and store metrics
        """

        # Calculate the metrics in both directions
        self._calculate_metrics_for_epoch(self.metrics.name, epoch, "B")
        self._calculate_metrics_for_epoch(f"{self.metrics.name}_INVERTED", epoch, "A")

        if epoch % self.plot_epochs == 0 and self.plot_image_epoch:
            plot_metrics(self.metrics_per_epoch)

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
