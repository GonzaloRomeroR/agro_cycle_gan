import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GANs to generate image synthetic datasets."
    )
    parser.add_argument(
        "use_dataset",
        type=str,
        help="name of the dataset to use",
        default=None,
    )
    parser.add_argument(
        "--download_dataset", help="download or obtain dataset", action="store_true"
    )
    parser.add_argument(
        "--image_resize",
        help="size of the image to resize",
        default=None,
        nargs=2,
        type=int,
    )
    parser.add_argument(
        "--tensorboard",
        help="generate tensorboard files",
        action="store_true",
    )
    parser.add_argument(
        "--store_models",
        action="store_true",
        help="store models in specific folder with datetime",
    )
    parser.add_argument(
        "--load_models",
        action="store_true",
        help="load the trained models from previous saves",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="batch size for the training process",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="obtain metrics every epoch",
        default="",
    )
    parser.add_argument(
        "--plot_image_epoch",
        help="plot images after every epoch",
        action="store_true",
    )
    parser.add_argument(
        "--generator",
        type=str,
        help="name of the generator to use",
        default="",
    )
    parser.add_argument(
        "--discriminator",
        type=str,
        help="name of the discriminator to use",
        default="",
    )
    parser.add_argument(
        "--comments",
        type=str,
        help="comments to add in results artifacts",
        default="",
    )
    parser.add_argument(
        "--db_connection_str",
        type=str,
        help="connection string to store results in a database",
        default="",
    )
    return parser.parse_args()
