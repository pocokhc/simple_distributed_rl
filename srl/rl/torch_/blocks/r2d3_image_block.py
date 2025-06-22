from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from srl.rl.torch_.converter import convert_activation_torch


class R2D3ImageBlock(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, ...],
        filters: int = 16,
        activation="ReLU",
    ):
        super().__init__()

        activation = convert_activation_torch(activation)

        in_ch = in_shape[-3]
        self.h_layers = nn.ModuleList(
            [
                _ResBlock(in_ch, filters, activation),
                _ResBlock(filters, filters * 2, activation),
                _ResBlock(filters * 2, filters * 2, activation),
                activation(),
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


class _ResBlock(nn.Module):
    def __init__(self, in_ch, filters, activation):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=filters,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.res1 = _ResidualBlock(filters, activation)
        self.res2 = _ResidualBlock(filters, activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class _ResidualBlock(nn.Module):
    def __init__(self, filters, activation):
        super().__init__()
        self.act1 = activation()
        self.conv1 = nn.Conv2d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.act2 = activation()
        self.conv2 = nn.Conv2d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.add = nn.Identity()

    def forward(self, x):
        x1 = self.act1(x)
        x1 = self.conv1(x1)
        x1 = self.act2(x1)
        x1 = self.conv2(x1)
        return self.add(x) + x1


if __name__ == "__main__":
    m = R2D3ImageBlock((96, 96, 3))
    print(m)
