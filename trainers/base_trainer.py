import time
from abc import ABC, abstractmethod

from utils.file_utils import save_models
from utils.metrics_utils import calculate_metrics
from utils.plot_utils import plot_generator_images
from utils.sys_utils import get_cpu_usage, get_gpu_usage, get_memory_usage
from utils.tensorboard_utils import TensorboardHandler


class BaseTrainer(ABC):
    def __init__(
        self,
        G_A2B,
        G_B2A,
        D_A,
        D_B,
        models_path,
        images_A,
        images_B,
        dataset_name,
        device,
        test_images_A,
        test_images_B,
        bs=5,
        num_epochs=20,
        plot_epochs=1,
        print_info=3,
        params_logger=None,
        metrics=None,
        im_size=(64, 64),
        tensorboard=False,
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
        :param images_A: dataloader with images of the domain A
        :type images_A: ´DataLoader´
        :param images_B: dataloader with images of the domain B
        :type images_B: ´DataLoader´
        :param models_path: path to load and save the models
        :type models_path: str
        :param dataset_name: name of the models
        :type dataset_name: str
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
        """
        self.G_A2B = G_A2B
        self.G_B2A = G_B2A
        self.D_A = D_A
        self.D_B = D_B
        self.models_path = models_path
        self.images_A = images_A
        self.images_B = images_B
        self.dataset_name = dataset_name
        self.device = device
        self.test_images_A = test_images_A
        self.test_images_B = test_images_B
        self.bs = bs
        self.num_epochs = num_epochs
        self.plot_epochs = plot_epochs
        self.print_info = print_info
        self.params_logger = params_logger
        self.metrics = metrics
        self.im_size = im_size
        self._define_storing()
        self.tensorboard = tensorboard
        if self.tensorboard:
            self._set_tensorboard()

    def _define_storing(self):
        self.D_A_losses = []
        self.D_B_losses = []
        self.G_losses = []

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

        self.losses_epoch = {key: [] for key in self.losses_names}
        self.losses_total = {key: [] for key in self.losses_names}

    def _set_tensorboard(self):
        self.writer = TensorboardHandler("./runs/Losses")

    def generate_images_cycle(self, a_real, b_real, G_A2B, G_B2A):
        """
        Create fake and reconstructed images.
        """
        b_fake = G_A2B(a_real)
        a_recon = G_B2A(b_fake)
        a_fake = G_B2A(b_real)
        b_recon = G_A2B(a_fake)
        return a_fake, b_fake, a_recon, b_recon

    def _print_iter_info(self, epoch, iteration, gen_losses, disc_losses):
        info = f"Epoch [{epoch}/{self.num_epochs}] batch [{iteration}]"
        for name, loss in {**gen_losses, **disc_losses}.items():
            info += f" {name}: {loss:.3f}"
        print(info)

    def _run_post_epoch(self, epoch):
        # Obtain total losses
        for key in self.losses_epoch.keys():
            loss_key = sum(self.losses_epoch[key]) / len(self.losses_epoch[key])
            self.losses_total[key].append(loss_key)
            if self.tensorboard:
                self.writer.add_scalar(key, loss_key, epoch)

        self.losses_epoch = {key: [] for key in self.losses_names}

        # Generate epoch information
        self.params_logger.params["final_epoch"] = epoch
        self.params_logger.params["time"] = time.perf_counter() - self.start_time
        self.params_logger.generate_params_file()

        save_models(
            self.models_path,
            self.dataset_name,
            self.G_A2B,
            self.G_B2A,
            self.D_A,
            self.D_B,
        )
        if epoch % self.plot_epochs == 0:
            plot_generator_images(
                self.G_A2B,
                self.G_B2A,
                self.test_images_A,
                self.test_images_B,
                self.device,
            )

        print(get_gpu_usage())
        print(get_memory_usage())
        print(get_cpu_usage())

        print(f"GPU Usage: {get_gpu_usage()}")
        print(f"Memory Usage: {get_memory_usage()}")
        print(f"CPU usage: {get_cpu_usage()}")

        print(
            "Epoch {}: GPU Usage {}, Memory Usage {} %, CPU usage {} %".format(
                epoch, get_gpu_usage(), get_memory_usage(), get_cpu_usage()
            )
        )
        # Obtain metrics
        if self.metrics:
            score = calculate_metrics(self.metrics, self.dataset_name, self.im_size[1:])
            print(f"{self.metrics.name} score: {score}")

    def train(self):
        self.start_time = time.perf_counter()
        self._train_model()
        end_time = time.perf_counter() - self.start_time
        print(f"Full trainig took {end_time} s to finish")
        return self.losses_total

    @abstractmethod
    def _train_model(self):
        pass
