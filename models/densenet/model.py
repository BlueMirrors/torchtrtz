"""Load DenseNet121 model and return it.

Returns:
    Module: Torch Module of DenseNet121.
"""
from torchvision.models import densenet121

from models.base_model import Model


class DenseNet(Model):
    """Loads DenseNet121 model.
    """
    def __init__(self) -> None:
        """Initialize densenet121 model.
        """
        #setup
        super().__init__(model=densenet121(pretrained=True))


if __name__ == "__main__":
    dn = DenseNet()
    print(dn.model)
