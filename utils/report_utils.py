import matplotlib.pyplot as plt
from pathlib import Path

from typing import Dict


def generate_loss_plot(losses: Dict):
    """Create loss plots

    :param losses: dictionary with the different losses stored during training
    :type losses: dict
    """
    Path("./results/losses_plots/").mkdir(parents=True, exist_ok=True)

    for loss_name in losses.keys():
        plt.plot(losses[loss_name])
        plt.grid()
        plt.title(loss_name)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.savefig(f"./results/losses_plots/{loss_name}.png")
        plt.close()


def generate_model_plots():
    pass


def generate_report(losses: Dict):
    generate_loss_plot(losses)
    generate_model_plots()

