import sys

sys.path.append("../")
sys.path.append("../../")
from train import train, Config


def test_training():
    config = Config()
    config.image_resize = [64, 64]
    config.use_dataset = "testA2testB"
    config.batch_size = 10
    config.num_epochs = 1
    train(config)

    assert True


def test_generation():
    assert True
