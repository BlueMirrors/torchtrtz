"""Load WideResnet model and return it.

Returns:
    Module: Torch Module of WideResnet.
"""
from torch.nn import Module
from torchsummary import summary
import torchvision.models as models


class WideResnet:
    """Loads WideResnet121 model.
    """
    def __init__(self) -> None:
        """Initialize WideResnet121 model.
        """
        self._model = models.wide_resnet50_2(pretrained=True)

        # Set model to eval mode
        self._model.cuda()
        self._model.eval()

    @property
    def model(self) -> Module:
        """Getter for the model

        Returns:
            Module: torch model
        """
        return self._model

    def print_summary(self) -> None:
        """Print summary of the model.
        """
        print(summary(self._model, input_size=(3, 224, 224)))


if __name__ == "__main__":
    dn = WideResnet()
    print(dn.model)
