from typing import Tuple

import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_size: int,
        layer_sizes: Tuple[int, ...] = (512,),
        activation="ReLU",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(activation, str):
            activation = getattr(nn, activation)

        self.layers = nn.ModuleList()
        for i, h in enumerate(layer_sizes):
            self.layers.append(
                nn.Linear(
                    in_features=layer_sizes[i - 1] if i > 0 else in_size,
                    out_features=h,
                    bias=True,
                )
            )
            self.layers.append(activation(inplace=True))

        # --- out shape
        self.out_shape = (layer_sizes[-1],)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
