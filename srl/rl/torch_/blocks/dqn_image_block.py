from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from srl.rl.torch_.converter import convert_activation_torch


class DQNImageBlock(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, ...],
        filters: int = 32,
        activation="ReLU",
    ):
        super().__init__()

        activation = convert_activation_torch(activation)

        in_ch = in_shape[-3]

        # (batch, in_ch, 84, 84)
        # -> (batch, 32, 21, 21)
        # -> (batch, 64, 11, 11)
        # -> (batch, 64, 11, 11)
        self.image_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_ch,
                    filters,
                    kernel_size=8,
                    stride=4,
                    padding=3,
                    padding_mode="replicate",
                ),
                activation(),
                nn.Conv2d(
                    filters,
                    filters * 2,
                    kernel_size=4,
                    stride=2,
                    padding=2,
                    padding_mode="replicate",
                ),
                activation(),
                nn.Conv2d(
                    filters * 2,
                    filters * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
                activation(),
            ]
        )

        # --- out shape
        x = np.ones((1,) + in_shape, dtype=np.float32)
        y = self.forward(torch.tensor(x))
        self.out_shape = y.shape[-3:]

    def forward(self, x):
        for layer in self.image_layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    m = DQNImageBlock((96, 72, 3))
    print(m)
