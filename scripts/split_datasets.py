import os
from pathlib import Path
from shutil import copyfile, rmtree
import random
import argparse


def list_splitter(list_to_split, ratio):
    """
    Randomly split list based on ratio
    """
    elements = len(list_to_split)
    random.shuffle(list_to_split)
    middle = int(elements * ratio)
    return list_to_split[:middle], list_to_split[middle:]


def split_dataset(
    folder_A: str,
    folder_B: str,
    output_name: str,
    dest_folder="./images/",
    perc_train=0.95,
):
    """
    Split image dataset for both domains in train and test.

    :param folder_A: path to the folder containing images of domain A
    :type folder_A: str
    :param folder_B: path to the folder containing images of domain B
    :type folder_B: str
    :param output_name: name of the folder containing the dataset
    :type output_name: str
    :param dest_folder: path where the folder will be created, defaults to ".images/"
    :type dest_folder: str, optional
    :param perc_train: proportion of data used to train, defaults to 0.95
    :type perc_train: float, optional
    """
    # Create folders
    dataset_path = f"{dest_folder}{output_name}"
    if Path(dataset_path).is_dir():
        rmtree(dataset_path)
    Path(dataset_path).mkdir(parents=True, exist_ok=True)
    for data in ["train", "test"]:
        for domain in ["A", "B"]:
            Path(f"{dataset_path}/{data}_{domain}").mkdir(parents=True, exist_ok=True)
            Path(f"{dataset_path}/{data}_{domain}/{domain}").mkdir(
                parents=True, exist_ok=True
            )
    for name, folder in {"A": folder_A, "B": folder_B}.items():
        train, test = list_splitter(os.listdir(folder), perc_train)
        for set_name, data in {"train": train, "test": test}.items():
            for image in data:
                copyfile(
                    f"{folder}/{image}",
                    f"{dataset_path}/{set_name}_{name}/{name}/{image}",
                )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Split dataset to be used in current image transformation"
    )
    parser.add_argument(
        "folder_A",
        type=str,
        help="path to the folder containing images of domain A",
        default=None,
    )
    parser.add_argument(
        "folder_B",
        type=str,
        help="path to the folder containing images of domain B",
        default=None,
    )
    parser.add_argument(
        "output_name",
        type=str,
        help="name of the folder to contain the dataset",
        default=None,
    )
    parser.add_argument(
        "--perc_train", type=int, help="proportion of data used to train", default=0.95,
    )
    parser.add_argument(
        "--dest_folder",
        type=str,
        help="proportion of data used to train",
        default="./images/",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cmd_args = parse_arguments()
    folder_A = cmd_args.folder_A
    folder_B = cmd_args.folder_B
    output_name = cmd_args.output_name
    perc_train = cmd_args.perc_train
    dest_folder = cmd_args.dest_folder
    split_dataset(folder_A, folder_B, output_name, dest_folder, perc_train)
