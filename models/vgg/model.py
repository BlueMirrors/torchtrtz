"""Load VGG16 model and return it.

Returns:
    Module: Torch Module of VGG16.
"""
import torchvision.models as models

from models.base_model import Model


class VGG16(Model):
    """Loads VGG16 model.
    """
    def __init__(self, batch_norm=False) -> None:
        """Initialize model based on batch normalization parameter.

        Args:
            batch_norm (bool, optional): Include batch normalization layer. Defaults to False.
        """
        # load model
        vgg16_model = models.vgg16_bn(
            pretrained=True) if batch_norm else models.vgg16(pretrained=True)

        # setup
        super().__init__(model=vgg16_model)
        self._bn = batch_norm


if __name__ == "__main__":
    vgg = VGG16()
    print(type(vgg.model))
