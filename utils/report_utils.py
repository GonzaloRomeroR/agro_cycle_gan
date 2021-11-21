import matplotlib.pyplot as plt
from pathlib import Path
from torchsummary import summary
from contextlib import redirect_stdout
import os

from typing import Dict
from fpdf import FPDF


class ParamsLogger:
    """
    Class to store the training params
    """

    _instance = None

    # Singleton to create only one param logger in the whole training process
    def __new__(class_, *args, **kwargs):
        if class_._instance is None:
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance

    def __init__(self):
        self.params = {}

    def generate_params_file(self):
        """Creates a txt with the params used
        """
        with open("./results/params.txt", "w") as f:
            with redirect_stdout(f):
                for param, value in self.params.items():
                    print(f"{param}: {value}")


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


def print_summary(model, size):
    summary(model, size)


def generate_model_file(G_A2B, G_B2A, D_A, D_B, size=(3, 64, 64)):
    """Creates a txt with the models used
    """
    with open("./results/models.txt", "w") as f:
        with redirect_stdout(f):
            print("Generator A TO B:")
            summary(G_A2B, size)
            print("\n\nGenerator B TO A:")
            summary(G_B2A, size)
            print("\n\nDiscriminator A:")
            summary(D_A, size)
            print("\n\nDiscriminator B:")
            summary(D_B, size)


def create_pdf():
    """
    Create pdf report
    """
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 20)
    pdf.cell(30, 10, "Training report", 0, 1)

    pdf.set_font("Arial", "B", 10)
    pdf.cell(30, 10, "Training parameters", 0, 1)

    with open("./results/params.txt", "r") as params_file:
        params_data = params_file.readlines()

    pdf.set_font("Arial", "", 8)
    for line in params_data:
        pdf.set_x(15)
        pdf.cell(25, 3, line, 0, 1)

    pdf.set_font("Arial", "B", 10)
    pdf.cell(30, 10, "Losses", 0, 1)
    for plot in os.listdir("./results/losses_plots"):
        pdf.image(f"./results/losses_plots/{plot}", w=100, h=70)

    pdf.set_font("Arial", "B", 10)
    pdf.cell(30, 10, "Models", 0, 1)

    with open("./results/models.txt", "r") as model_file:
        model_data = model_file.readlines()

    pdf.set_font("Arial", "", 6)
    for line in model_data:
        pdf.set_x(25)
        if "Discriminator" in line or "Generator" in line:
            pdf.set_font("Arial", "B", 6)
            pdf.cell(25, 3, line, 0, 1)
            pdf.set_font("Arial", "", 6)
        else:
            pdf.cell(25, 3, line, 0, 1)

    pdf.output("./results/report.pdf", "F")


def generate_report(losses: Dict):
    generate_loss_plot(losses)
    generate_model_plots()
    create_pdf()

