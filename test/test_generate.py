"""Contains tests for all the models.
"""
import sys
import pytest

sys.path.insert(0, './')

from generate_weights import load_model

# list of models to test
MODELS = ['VGG16', 'DenseNet121', 'AlexNet', 'Inceptionv4', 'WideResnet']


@pytest.mark.parametrize('model_name', MODELS)
def test_conversion(model_name):
    """Load model and convert

    Args:
        model_name (str): Name of the model to convert weights of
    """
    model = load_model(model_name)
    model.generate_weights('temp.wts')
    assert True, f"Failed to generate for {model_name}"
