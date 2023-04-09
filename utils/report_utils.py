import os
import shutil
import json
import re
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from fpdf import FPDF
from models.discriminators.base_discriminator import BaseDiscriminator
from models.generators.base_generator import BaseGenerator
from torchsummary import summary


class ParamsLogger:
    """
    Class to store the training params
    """

    _instance = None

    # Singleton to create only one param logger in the whole training process
    def __new__(class_, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        if class_._instance is None:
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance

    def __init__(self) -> None:
        if not hasattr(self, "params"):
            self.params: Dict[str, Any] = {}

    def generate_params_file(self) -> None:
        """Creates a txt with the params used"""
        file_path = Path(__file__).parent.resolve()
        with open(f"{file_path}/../results/params.txt", "w") as f:
            with redirect_stdout(f):
                for param, value in self.params.items():
                    print(f"{param}: {value}")

    def set_params(self, params_dict: Dict[str, Any]) -> None:
        self.params.update(params_dict)


def generate_loss_plot(losses: Dict[str, Any]) -> None:
    """Create loss plots

    :param losses: dictionary with the different losses stored during training
    :type losses: dict
    """
    file_path = Path(__file__).parent.resolve()
    Path(f"{file_path}/../results/losses_plots/").mkdir(parents=True, exist_ok=True)

    for loss_name in losses.keys():
        plt.figure()
        plt.plot(losses[loss_name])
        plt.grid()
        plt.title(loss_name)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.savefig(f"{file_path}/../results/losses_plots/{loss_name}.png")
        plt.close()


def generate_metrics_plot(metrics: Dict[str, Any]) -> None:
    """Create metrics plots

    :param metrics: list with the metrics stored during training
    :type metrics: dict
    """
    file_path = Path(__file__).parent.resolve()
    Path(f"{file_path}/../results/metrics_plots/").mkdir(parents=True, exist_ok=True)

    for metrics_name in metrics.keys():
        plt.figure()
        plt.plot(metrics[metrics_name])
        plt.grid()
        plt.title(metrics_name)
        plt.xlabel("epochs")
        plt.ylabel(metrics_name)
        plt.savefig(f"{file_path}/../results/metrics_plots/{metrics_name}.png")
        plt.close()


def generate_report_json(
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    losses: Dict[str, Any],
    models_path="./results/models.txt",
):

    """Create json with all the useful information obtained during the training

    :param params: dictionary with the values stored in the parameter logger
    :type params: dict
    :param metrics: dictionary with the different metrics
    :type metrics: dict
    :param losses: dictionary with the different losses
    :type losses: dict
    :param models_path: path to store the generated json
    :type models_path: str

    """
    results = {}
    results.update(params)

    metrics_to_list = {}
    for name, metrics in metrics.items():
        metrics_to_list[name] = [metric.item() for metric in metrics]

    results["metrics"] = metrics_to_list
    results["losses"] = losses

    if os.path.isfile(models_path):
        with open(models_path) as f:
            results["models"] = f.readlines()

    output_name = "train"
    if "dataset" in results:
        output_name += f"_{results['dataset']}"
    if "date" in results:
        output_name += f"_{results['date']}"

    with open(
        "./results/" + re.sub(r"[^\w\-_\. ]", "_", f"{output_name}.json"), "w"
    ) as fp:
        json.dump(results, fp)


def generate_model_file(
    G_A2B: BaseGenerator,
    G_B2A: BaseGenerator,
    D_A: BaseDiscriminator,
    D_B: BaseDiscriminator,
    size: Tuple[int, ...] = (3, 64, 64),
) -> None:
    """Creates a txt with the models used"""
    file_path = Path(__file__).parent.resolve()
    with open(f"{file_path}/../results/models.txt", "w") as f:
        with redirect_stdout(f):
            print("Generator A TO B:")
            summary(G_A2B, size)
            print("\n\nGenerator B TO A:")
            summary(G_B2A, size)
            print("\n\nDiscriminator A:")
            summary(D_A, size)
            print("\n\nDiscriminator B:")
            summary(D_B, size)


def create_pdf() -> None:
    """
    Create pdf report
    """
    file_path = Path(__file__).parent.resolve()
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 20)
    pdf.cell(30, 10, "Training report", 0, 1)

    pdf.set_font("Arial", "B", 10)
    pdf.cell(30, 10, "Training parameters", 0, 1)

    with open(f"{file_path}/../results/params.txt", "r") as params_file:
        params_data = params_file.readlines()

    pdf.set_font("Arial", "", 8)
    for line in params_data:
        pdf.set_x(15)
        pdf.cell(25, 3, line, 0, 1)

    pdf.set_font("Arial", "B", 10)
    pdf.cell(30, 10, "Losses", 0, 1)
    for plot in os.listdir(f"{file_path}/../results/losses_plots"):
        pdf.image(f"{file_path}/../results/losses_plots/{plot}", w=100, h=70)

    pdf.set_font("Arial", "B", 10)
    pdf.cell(30, 10, "Metrics", 0, 1)
    for plot in os.listdir(f"{file_path}/../results/metrics_plots"):
        pdf.image(f"{file_path}/../results/metrics_plots/{plot}", w=100, h=70)

    pdf.set_font("Arial", "B", 10)
    pdf.cell(30, 10, "Models", 0, 1)

    with open(f"{file_path}/../results/models.txt", "r") as model_file:
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

    file_path = Path(__file__).parent.resolve()
    pdf.output(f"{file_path}/../results/report.pdf", "F")


def clear_folder(folder: str) -> None:

    """Clear results folder"""
    if not os.path.isdir(folder):
        return

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def clear_reports() -> None:
    """Clear old reports"""
    file_path = Path(__file__).parent.resolve()

    if os.path.isfile(f"{file_path}/../results/report.pdf"):
        os.remove(f"{file_path}/../results/report.pdf")

    clear_folder(f"{file_path}/../results/losses_plots")
    clear_folder(f"{file_path}/../results/metrics_plots")


def generate_report(
    train_params: Dict[str, Any], losses: Dict[str, Any], metrics: Dict[str, Any]
) -> None:

    """Generate training report"""
    clear_reports()
    generate_loss_plot(losses)

    if bool(metrics):
        generate_metrics_plot(metrics)

    generate_report_json(train_params, metrics, losses)
    create_pdf()
