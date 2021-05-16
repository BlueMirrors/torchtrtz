"""Contains tests for all the models.
"""
import sys

sys.path.insert(0, './')

from generate_weights import load_model


def generate(model_name: str):
    """Load model and convert

    Args:
        model_name (str): name of the model
    """
    model = load_model(model_name)
    model.generate_weights('temp.wts')
    return True


def test_vgg16():
    """Test VGG16 Model Conversion
    """
    model_name = "VGG16"
    assert generate(model_name), \
        f"Failed to generate for {model_name}"


def test_densenet121():
    """Test DenseNet121 Model Conversion
    """
    model_name = "DenseNet121"
    assert generate(model_name), \
        f"Failed to generate for {model_name}"


def test_alexnet():
    """Test AlexNet Model Conversion
    """
    model_name = "AlexNet"
    assert generate(model_name), \
        f"Failed to generate for {model_name}"


def test_inceptionv4():
    """Test Inceptionv4 Model Conversion
    """
    model_name = "Inceptionv4"
    assert generate(model_name), \
        f"Failed to generate for {model_name}"


def test_wide_resnet():
    """Test WideResnet Model Conversion
    """
    model_name = "WideResnet"
    assert generate(model_name), \
        f"Failed to generate for {model_name}"