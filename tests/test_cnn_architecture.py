
from src.models.cnn import (
    CHANNELS,
    INPUT_SIZE,
    NUM_CLASSES,
    Custom10LayerCNN,
)


def test_cnn_architecture_configuration():
    model = Custom10LayerCNN()
    config = model.architecture_config()

    assert config["input_channels"] == 1
    assert config["input_size"] == list(INPUT_SIZE)
    assert config["channels"] == list(CHANNELS)
    assert config["num_classes"] == NUM_CLASSES
    assert len(config["channels"]) == 10
    assert config["kernel_size"] == 3
    assert config["padding"] == 1
    assert config["activation"] == "ReLU"


def test_cnn_uses_expected_pooling_and_classifier():
    model = Custom10LayerCNN()
    config = model.architecture_config()

    assert "2, 4, 6, and 8" in config["pooling"]
    assert config["global_pooling"] == (
        "Adaptive Average Pooling to 1x1"
    )
    assert config["classifier"] == "one linear layer"


def test_cnn_has_two_output_classes():
    model = Custom10LayerCNN()

    assert model.num_classes == 2
    assert model.architecture_config()["num_classes"] == 2
