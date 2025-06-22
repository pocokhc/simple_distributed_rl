from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from srl.rl.torch_.converter import convert_activation_torch


class MuZeroAtariBlock(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        filters: int = 128,
        activation="ReLU",
        use_layer_normalization: bool = False,
    ):
        super().__init__()

        activation = convert_activation_torch(activation)

        in_ch = in_shape[-3]
        self.h_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=filters,
                    kernel_size=(3, 3),
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                activation(),
                _ResidualBlock(filters, activation, use_layer_normalization),
                _ResidualBlock(filters, activation, use_layer_normalization),
                nn.Conv2d(
                    in_channels=filters,
                    out_channels=filters * 2,
                    kernel_size=(3, 3),
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                activation(),
                _ResidualBlock(filters * 2, activation, use_layer_normalization),
                _ResidualBlock(filters * 2, activation, use_layer_normalization),
                _ResidualBlock(filters * 2, activation, use_layer_normalization),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                _ResidualBlock(filters * 2, activation, use_layer_normalization),
                _ResidualBlock(filters * 2, activation, use_layer_normalization),
                _ResidualBlock(filters * 2, activation, use_layer_normalization),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )

        # --- out shape
        x = np.ones((1,) + in_shape, dtype=np.float32)
        y = self.forward(torch.tensor(x))
        self.out_shape = y.shape[-3:]

    def forward(self, x):
        for layer in self.h_layers:
            x = layer(x)
        return x


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        filters: int,
        activation,
        use_layer_normalization: bool,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=(3, 3),
            padding=1,
            bias=False,
        )
        if use_layer_normalization:
            self.bn1 = nn.LayerNorm(filters)
        else:
            self.bn1 = nn.BatchNorm2d(filters)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=(3, 3),
            padding=1,
            bias=False,
        )
        if use_layer_normalization:
            self.bn2 = nn.LayerNorm(filters)
        else:
            self.bn2 = nn.BatchNorm2d(filters)
        self.act2 = activation()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.act2(x)
        return x


if __name__ == "__main__":
    m = MuZeroAtariBlock((96, 72, 3))
    print(m)
