"""Load WideResnet model and return it.

Returns:
    Module: Torch Module of WideResnet.
"""
from torchvision.models import wide_resnet50_2

from models.base_model import Model


class WideResnet(Model):
    """Loads WideResnet121 model.
    """
    def __init__(self) -> None:
        """Initialize WideResnet121 model.
        """
        super().__init__(model=wide_resnet50_2(pretrained=True))


if __name__ == "__main__":
    dn = WideResnet()
    print(dn.model)
