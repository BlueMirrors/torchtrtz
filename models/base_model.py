"""Contains class def of Model.
"""
from typing import Tuple
import struct

import torch
from torch.nn import Module
from torchsummary import summary


class Model:
    """Base Model class that implements
    common methods of all models.
    """
    def __init__(
        self, model: Module, input_size: Tuple[int] = (3, 224, 224)) -> None:
        """Initialize Model.

        Args:
            model ([type]): [description]
            input_size (tuple, optional): Input size of model. Defaults to (3, 224, 224).

        Raises:
            Exception: Raised if cuda is not available.
        """
        self._model = model

        # cuda is required for tensorrt
        if not torch.cuda.is_available():
            raise Exception('Cuda not available.')

        # Set model to eval mode,
        self._model.cuda()
        self._model.eval()

        # set input size for later use
        self._input_size = input_size

    @property
    def model(self) -> Module:
        """Getter for the model

        Returns:
            Module: torch model
        """
        return self._model

    @property
    def input_size(self) -> Tuple[int]:
        """Getter for input_size

        Returns:
            Tuple[int]: Input size of model
        """
        return self._input_size

    def print_summary(self) -> None:
        """Print summary of the model.
        """
        print(summary(self._model, input_size=self._input_size))

    def generate_weights(self, trt_weights_path: str) -> None:
        """Convert torch weights format to wts weights format

        Args:
            trt_weights_path (str): Path where trt weights will be saved.
        """
        # open tensorrt weights file
        wts_file = open(trt_weights_path, "w")

        # write length of keys
        print("Keys: ", self._model.state_dict().keys())
        wts_file.write("{}\n".format(len(self._model.state_dict().keys())))
        for key, val in self._model.state_dict().items():
            print("Key: {}, Val: {}".format(key, val.shape))
            vval = val.reshape(-1).cpu().numpy()
            wts_file.write("{} {}".format(key, len(vval)))
            for v_l in vval:
                wts_file.write(" ")

                # struct.pack Returns a bytes object containing the values v1, v2, …
                # packed according to the format string format (>big endian in this case).
                wts_file.write(struct.pack(">f", float(v_l)).hex())
            wts_file.write("\n")

        wts_file.close()
