import sys

from train import train, Config
from generate import ImageTransformer
from pathlib import Path

file_path = Path(__file__).parent.resolve()


def test_training():
    config = Config()
    config.image_resize = [64, 64]
    config.use_dataset = "testA2testB"
    config.batch_size = 10
    config.num_epochs = 1
    train(config)

    assert True


def test_generation():
    image_transformer = ImageTransformer("testA2testB")
    image_transformer.transform_dataset(
        origin_path=f"{file_path}/../../images/testA2testB/test_A/A",
        dest_path=f"{file_path}/../../images_gen/",
    )
    assert True
