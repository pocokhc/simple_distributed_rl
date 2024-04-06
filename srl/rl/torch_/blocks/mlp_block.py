from typing import Tuple

import torch.nn as nn

from srl.rl.torch_.converter import convert_activation_torch, set_initializer_torch
from srl.rl.torch_.modules.noisy_linear import NoisyLinear


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_size: int,
        layer_sizes: Tuple[int, ...] = (512,),
        activation="ReLU",
        use_bias=True,
        kernel_initializer: str = "he_normal",
        bias_initializer="zeros",
        enable_noisy_dense: bool = False,
    ):
        super().__init__()

        activation = convert_activation_torch(activation)

        self.hidden_layers = nn.ModuleList()
        for i in range(len(layer_sizes)):
            if enable_noisy_dense:
                m = NoisyLinear(in_size, layer_sizes[i])
            else:
                m = nn.Linear(in_size, layer_sizes[i], bias=use_bias)
                if kernel_initializer != "":
                    set_initializer_torch(m.weight, kernel_initializer)
                if use_bias and bias_initializer != "":
                    set_initializer_torch(m.bias, bias_initializer)
            self.hidden_layers.append(m)
            self.hidden_layers.append(activation(inplace=True))
            in_size = layer_sizes[i]

        # --- out shape
        self.out_size = in_size

    def add_layer(self, layer, out_size):
        self.hidden_layers.append(layer)
        self.out_size = out_size

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x
