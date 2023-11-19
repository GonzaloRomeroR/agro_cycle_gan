import shutil

from datetime import datetime
from time import gmtime, strftime
from dacite import from_dict

from trainers.basic_trainer import BasicTrainer, BaseTrainer
from utils.file_utils import download_images, get_models
from utils.image_utils import datasets_get
from utils.parser_utils import parse_arguments
from utils.report_utils import ParamsLogger, ResultsReporter
from utils.sys_utils import get_device, system_configuration
from utils.tensorboard_utils import create_models_tb
from utils.train_utils import Config


def train(config: Config) -> None:
    """
    Train model saving information about the output
    """
    dataset_name = config.use_dataset

    trainer = _setup_trainer(config)
    losses = trainer.train()

    if config.store_models:
        shutil.copytree(
            f"./results/{dataset_name}",
            f"./results/{dataset_name}/models_{str(strftime('%Y-%m-%d-%H:%M:%S', gmtime()))}",
        )

    # Generate report
    ResultsReporter.generate_report(
        dataset_name,
        ParamsLogger().params,
        losses,
        trainer.metrics_per_epoch,
        config.db_connection_str,
    )

    ResultsReporter.generate_example_images(dataset_name)



def _setup_trainer(config: Config) -> BaseTrainer:
    """
    Setup initial configuration and return trainer
    """
    print("Performing setup")
    system_configuration()

    device = get_device(debug=True)
    dataset_name = config.use_dataset
    image_resize = config.image_resize

    # Get images
    if config.download_dataset:
        download_images(dataset_name)

    # Set image size
    if image_resize:
        # Do this to avoid mypy warnings
        im_size_list = [int(value) for value in [3] + image_resize]
        im_size = tuple(im_size_list)
    else:
        im_size = (3, 64, 64)

    # Get datasets from both domains
    batch_size = config.batch_size
    datasets = datasets_get(dataset_name, im_size[1:], batch_size)

    # Create or load models for training
    G_A2B, G_B2A, D_A, D_B = get_models(
        dataset_name,
        device,
        config.load_models,
        disc_name=config.discriminator,
        gen_name=config.generator,
    )

    # Set tesorboard functions
    if config.tensorboard:
        images, _ = next(iter(datasets.train_A))
        create_models_tb(G_A2B, G_B2A, D_A, D_B, images.to(device))

    # Generate file with the model information
    ResultsReporter.generate_model_file(G_A2B, G_B2A, D_A, D_B, size=im_size)

    # Define params for report
    params_logger = ParamsLogger()
    domains = config.use_dataset.split("2")
    log_params = {
        "comments": config.comments,
        "date": str(datetime.now()),
        "dataset": config.use_dataset,
        "image_size": im_size,
        "batch_size": batch_size,
        "configured_epochs": config.num_epochs,
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
        datasets,
        config=config,
        device=device,
        params_logger=params_logger,
        im_size=im_size,
    )
    return trainer

if __name__ == "__main__":
    config = from_dict(data_class=Config, data=vars(parse_arguments()))
    train(config)
