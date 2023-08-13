from typing import Tuple

import torch.nn as nn

from srl.rl.models.converter import convert_activation_torch

from .noisy_dense import GaussianNoise


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_size: int,
        layer_sizes: Tuple[int, ...] = (512,),
        activation: str = "ReLU",
        enable_noisy_dense: bool = False,
    ):
        super().__init__()

        activation = convert_activation_torch(activation)

        self.layers = nn.ModuleList()
        for i, h in enumerate(layer_sizes):
            self.layers.append(
                nn.Linear(
                    in_features=layer_sizes[i - 1] if i > 0 else in_size,
                    out_features=h,
                    bias=True,
                )
            )
            if enable_noisy_dense:
                self.layers.append(GaussianNoise())
            self.layers.append(activation(inplace=True))

        # --- out shape
        if len(layer_sizes) == 0:
            self.out_size = in_size
        else:
            self.out_size = layer_sizes[-1]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
