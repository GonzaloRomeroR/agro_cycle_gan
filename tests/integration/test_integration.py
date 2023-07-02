from train import train, setup_trainer, Config
from generate import ImageTransformer
from pathlib import Path

file_path = Path(__file__).parent.resolve()


def test_training():
    """
    This test only checks that a training can be done
    without crashing with a simple image example
    """
    config = Config()
    config.image_resize = [64, 64]
    config.use_dataset = "testA2testB"
    config.batch_size = 10
    config.num_epochs = 1
    train(config)

    assert True


def test_generation():
    """
    This test only checks that a generation can be done
    without crashing with a simple image example
    """
    image_transformer = ImageTransformer("testA2testB")
    image_transformer.transform_dataset(
        origin_path=f"{file_path}/../../images/testA2testB/test_A/A",
        dest_path=f"{file_path}/../../images_gen/",
    )
    assert True


def test_metrics_decreasing():
    """
    Test that checks that the metrics decrease during training
    valid for the first epochs
    """

    config = Config()
    config.image_resize = [64, 64]
    config.use_dataset = "testA2testB"
    config.batch_size = 10
    config.num_epochs = 4

    trainer = setup_trainer(config)
    trainer.train()

    metric_values = trainer.metrics_per_epoch[next(iter(trainer.metrics_per_epoch))]

    assert metric_values[0].item() > metric_values[-1].item()
