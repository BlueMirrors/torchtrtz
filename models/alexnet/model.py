"""Load AlexNet model and return it.

Returns:
    Module: Torch Module of AlexNet.
"""
from torchvision.models import alexnet

from models.base_model import Model


class AlexNet(Model):
    """Loads AlexNet model.
    """
    def __init__(self) -> None:
        """Initialize AlexNet model.
        """
        # setup
        super().__init__(model=alexnet(pretrained=False))


if __name__ == "__main__":
    alex = AlexNet()
    print(alex.print_summary())
