import pathlib
import os
import difflib
import json
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join


class ResultsComparator:

    """
    Comparates different training results based on the json output files
    """

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def generate_files(self) -> None:

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.output_files = [
            f"{self.input_path}/{name}"
            for name in listdir(self.input_path)
            if isfile(join(self.input_path, name)) and "train_" in name
        ]

        self.output_dicts = {
            name: json.load(open(name, "r")) for name in self.output_files
        }

        self._generate_fid_plot()
        self._generate_losses_plots()
        self._generate_diff_file()

    def _generate_fid_plot(self) -> None:

        metrics_names = self.output_dicts[next(iter(self.output_dicts))][
            "metrics"
        ].keys()

        for metric in metrics_names:
            plt.figure()
            for _, data in self.output_dicts.items():
                plt.plot(data["metrics"][metric], label=data["comments"])
            plt.grid()
            plt.title(metric)
            plt.legend()
            plt.xlabel("epochs")
            plt.ylabel(metric)
            plt.savefig(f"{self.output_path}/{metric}.png")
            plt.close()

    def _generate_losses_plots(self) -> None:
        losses_names = self.output_dicts[next(iter(self.output_dicts))]["losses"].keys()

        for loss in losses_names:
            plt.figure()
            for _, data in self.output_dicts.items():
                plt.plot(data["losses"][loss], label=data["comments"])
            plt.grid()
            plt.title(loss)
            plt.legend()
            plt.xlabel("epochs")
            plt.ylabel(loss)
            plt.savefig(f"{self.output_path}/{loss}.png")
            plt.close()

    def _generate_diff_file(self) -> None:

        if len(self.output_dicts) < 2:
            return

        # Get two first results to compare
        diff_list = list(
            {key: self.output_dicts[key] for key in list(self.output_dicts)[:2]}.items()
        )

        with open(f"{self.output_path}/model_diff.txt", "w+") as file:
            for line in difflib.unified_diff(
                diff_list[0][1]["models"],
                diff_list[1][1]["models"],
                fromfile=diff_list[0][0],
                tofile=diff_list[1][0],
                lineterm="",
            ):
                file.write(line + "\n")


if __name__ == "__main__":
    file_path = pathlib.Path(__file__).parent.resolve()
    input_path = f"{file_path}/../results/"
    output_path = f"{file_path}/../results/comparison"
    comparator = ResultsComparator(input_path, output_path)
    comparator.generate_files()
