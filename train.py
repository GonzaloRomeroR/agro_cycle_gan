import argparse
from datetime import datetime

from trainers.basic_trainer import BasicTrainer
from utils.file_utils import download_images, get_models
from utils.image_utils import datasets_get
from utils.metrics_utils import FID
from utils.parser_utils import parse_arguments
from utils.report_utils import ParamsLogger, generate_model_file, generate_report
from utils.sys_utils import get_device, suppress_qt_warnings, suppress_sklearn_errors
from utils.tensorboard_utils import create_models_tb


def setup(cmd_args: argparse.Namespace) -> None:
    print("Performing setup")

    # Get images
    device = get_device(debug=True)
    dataset_name = cmd_args.use_dataset
    if cmd_args.download_dataset:
        print(f"Downloading dataset {dataset_name}")
        download_images(dataset_name)

    # Set image size
    image_resize = cmd_args.image_resize
    if image_resize:
        im_size = tuple([3] + image_resize)
    else:
        im_size = (3, 64, 64)

    # Get datasets from both domains
    batch_size = cmd_args.batch_size
    train_images_A, train_images_B, test_images_A, test_images_B = datasets_get(
        dataset_name, im_size[1:], batch_size
    )

    # Create or load models for training
    G_A2B, G_B2A, D_A, D_B = get_models(
        dataset_name,
        device,
        cmd_args.load_models,
        disc_name=cmd_args.discriminator,
        gen_name=cmd_args.generator,
    )

    # Set tesorboard functions
    if cmd_args.tensorboard:
        images, _ = next(iter(train_images_A))
        create_models_tb(G_A2B, G_B2A, D_A, D_B, images.to(device))

    # Create metrics object
    metrics = FID() if cmd_args.metrics else None

    # Generate file with the model information
    generate_model_file(G_A2B, G_B2A, D_A, D_B, size=im_size)

    # Define params for report
    params_logger = ParamsLogger()
    domains = cmd_args.use_dataset.split("2")
    log_params = {
        "date": datetime.now(),
        "dataset": cmd_args.use_dataset,
        "image_size": im_size,
        "batch_size": batch_size,
        "configured_epochs": cmd_args.num_epochs,
    }
    if len(domains) == 2:
        # The dataset name format is domainA2domainB
        log_params["domain_A"] = domains[0]
        log_params["domain_B"] = domains[1]

    for name, model in {"G_A2B": G_A2B, "G_B2A": G_B2A, "D_A": D_A, "D_B": D_B}.items():
        log_params[f"model_name_{name}"] = model.__class__.__name__
        log_params[f"learning_rate_{name}"] = model.lr
    params_logger.set_params(log_params)

    trainer = BasicTrainer(
        G_A2B,
        G_B2A,
        D_A,
        D_B,
        f"./results/{dataset_name}",
        train_images_A,
        train_images_B,
        dataset_name,
        num_epochs=cmd_args.num_epochs,
        device=device,
        test_images_A=test_images_A,
        test_images_B=test_images_B,
        bs=cmd_args.batch_size,
        params_logger=params_logger,
        metrics=metrics,
        im_size=im_size,
        tensorboard=cmd_args.tensorboard,
    )

    losses = trainer.train()

    # Generate report
    generate_report(losses)


def configure() -> None:
    suppress_sklearn_errors()
    suppress_qt_warnings()


if __name__ == "__main__":
    configure()
    cmd_args = parse_arguments()
    setup(cmd_args)

