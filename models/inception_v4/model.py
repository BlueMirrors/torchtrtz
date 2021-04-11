import json
import torch.utils.model_zoo as model_zoo
from torch import nn

import inception


class InceptionV4:
    """Loads Inceptionv4 model.
    """
    def __init__(self, num_classes=1000, config='config.json', pretrained='imagenet') -> None:
        """Initialize inception v4 model
        """
        self._model = None
        
        # load config
        with open(config) as f:
            self._config = json.load(f)

        self._load_model(num_classes, pretrained)
        
        # Set model to eval mode
        self._model.cuda()
        self._model.eval()

    @property
    def model(self) -> nn.Module:
        """Getter for the model

        Returns:
            nn.Module: torch model
        """
        return self._model

    def print_summary(self) -> None:
        """Print summary of the model.
        """
        print(summary(self._model, input_size=(3, 299, 299)))

    def _load_model(self, num_classes: int, pretrained: str) -> None:
        """Creates the inceptionv4 model and loads state dict.

        Args:
            num_classes (int): number of classes in the classifier
            pretrained (bool): If True, returns a model pre-trained on ImageNet. 
        """
        if pretrained:
            settings = self._config['inceptionv4'][pretrained]

            if num_classes != settings['num_classes']:
                raise ValueError(f'num_classes should be {settings["num_classes"]}, but is {num_classes}')

            # both 'imagenet'&'imagenet+background' are loaded from same parameters
            self._model = inception.Inception_v4(num_classes=1001)
            self._model.load_state_dict(model_zoo.load_url(settings['url']))

            if pretrained == 'imagenet':
                new_last_linear = nn.Linear(1536, 1000)
                new_last_linear.weight.data = self._model.last_linear.weight.data[1:]
                new_last_linear.bias.data = self._model.last_linear.bias.data[1:]
                self._model.last_linear = new_last_linear

            self._model.input_space = settings['input_space']
            self._model.input_size = settings['input_size']
            self._model.input_range = settings['input_range']
            self._model.mean = settings['mean']
            self._model.std = settings['std']
        else:
            self._model = inception.Inception_v4(num_classes=num_classes)


if __name__ == '__main__':
    net = InceptionV4(num_classes=1000, pretrained='imagenet')
    print(net.model)
    