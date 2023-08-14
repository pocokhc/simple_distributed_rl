from typing import Tuple

import torch.nn as nn

from srl.rl.models.converter import convert_activation_torch
from srl.rl.models.torch_.noisy_linear import NoisyLinear


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_size: int,
        layer_sizes: Tuple[int, ...] = (512,),
        activation="ReLU",
        enable_noisy_dense: bool = False,
    ):
        super().__init__()

        activation = convert_activation_torch(activation)

        if enable_noisy_dense:
            _Linear = NoisyLinear
        else:
            _Linear = nn.Linear

        self.hidden_layers = nn.ModuleList()
        for i in range(len(layer_sizes)):
            self.hidden_layers.append(_Linear(in_size, layer_sizes[i]))
            self.hidden_layers.append(activation(inplace=True))
            in_size = layer_sizes[i]

        # --- out shape
        self.out_size = in_size

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x
