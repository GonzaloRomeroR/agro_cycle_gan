import argparse
import subprocess
from typing import List

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create class diagrams for packages")

    parser.add_argument(
        "--package_folders",
        type=str,
        nargs="+",
        help="path to the folder containing images to crop",
        default=None,
    )
    parser.add_argument(
        "--dest_folder",
        type=str,
        help="path to the folder where the cropped images will be stored",
        default=None,
    )
    return parser.parse_args()

def create_class_diagram(folders: List[str] , destination: str):
    """
    Create class diagram for specific package

    :param folders: packages folders to generate the diagrams from
    :type folders: List[str]
    :param destination: path to store the generated diagrams
    :type destination: str
    """
    for folder in folders:
        subprocess.run(f"pyreverse -o png {folder}".split())
        subprocess.run(f"mv classes.png {destination}/{folder}.png".split())

if __name__ == "__main__":
    cmd_args = parse_arguments()
    create_class_diagram(cmd_args.package_folders, cmd_args.dest_folder)
