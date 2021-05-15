import json
import torch.utils.model_zoo as model_zoo
from torch import nn

import inception


class InceptionV4:
    """Loads Inceptionv4 model.
    """
    def __init__(self,
                 num_classes=1000,
                 config='config.json',
                 pretrained='imagenet') -> None:
        """Initialize inception v4 model
        """
        # load config
        config = None
        with open(config) as f:
            config = json.load(f)

        inception_model = self.load_model(num_classes, pretrained, config)

        super().__init__(model=inception_model, input_size=(3, 299, 299))

    @staticmethod
    def load_model(self, num_classes: int, pretrained: str,
                   config: dict) -> inception.Inception_v4:
        """Creates the inceptionv4 model and loads state dict.

        Args:
            num_classes (int): number of classes in the classifier
            pretrained (str): If True, returns a model pre-trained on ImageNet.
            config (dict): Configurations for setting up model.
        Raises:
            ValueError: raised if num classes don't match with the settings.

        Returns:
            inception.Inception_v4: loaded Inception model
        """

        model = None
        if pretrained:
            settings = config['inceptionv4'][pretrained]

            if num_classes != settings['num_classes']:
                raise ValueError(
                    f'num_classes should be {settings["num_classes"]}, but is {num_classes}'
                )

            # both 'imagenet'&'imagenet+background' are loaded from same parameters
            model = inception.Inception_v4(num_classes=1001)
            model.load_state_dict(model_zoo.load_url(settings['url']))

            if pretrained == 'imagenet':
                new_last_linear = nn.Linear(1536, 1000)
                new_last_linear.weight.data = model.last_linear.weight.data[1:]
                new_last_linear.bias.data = model.last_linear.bias.data[1:]
                model.last_linear = new_last_linear

            model.input_space = settings['input_space']
            model.input_size = settings['input_size']
            model.input_range = settings['input_range']
            model.mean = settings['mean']
            model.std = settings['std']
        else:
            model = inception.Inception_v4(num_classes=num_classes)

        return model


if __name__ == '__main__':
    net = InceptionV4(num_classes=1000, pretrained='imagenet')
    print(net.model)
