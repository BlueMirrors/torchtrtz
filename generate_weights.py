"""Python file to generate wts format of weights from torch models

Returns:
    weights.wts: Saves the weights of the model.
"""
import argparse

from models.base_model import Model


def load_model(model_name: str) -> Model:
    """Load model based on given model_name

    Args:
        model_name (str): Name of the model to be loaded

    Returns:
        model (Model): Loaded Model Object
    """
    if model_name == "VGG16":
        from models.vgg import VGG16
        model = VGG16()
    elif model_name == "DenseNet121":
        from models.densenet import DenseNet
        model = DenseNet()
    elif model_name == "AlexNet":
        from models.alexnet import AlexNet
        model = AlexNet()
    elif model_name == "Inceptionv4":
        from models.inception_v4 import InceptionV4
        model = InceptionV4()
    elif model_name == "WideResnet":
        from models.resnet import WideResnet
        model = WideResnet()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        help=
        'state the model name (along with layer information) based on the README file'
    )
    parser.add_argument('--save-trt-weights',
                        type=str,
                        default='vgg16.wts',
                        help='save path for TensorRT weights')
    args = parser.parse_args()

    dl_model = load_model(args.model)
    dl_model.generate_weights(args.save_trt_weights)
