from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from srl.rl.torch_.converter import convert_activation_torch


class AlphaZeroImageBlock(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        n_blocks: int = 19,
        filters: int = 256,
        activation="ReLU",
        **kwargs,
    ):
        super().__init__(**kwargs)

        activation = convert_activation_torch(activation)

        in_ch = in_shape[0]
        self.conv1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=filters,
            kernel_size=(3, 3),
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(filters)
        self.act1 = activation(inplace=True)

        self.resblocks = nn.ModuleList([_ResidualBlock(filters, filters, activation) for _ in range(n_blocks)])

        # --- out shape
        x = np.ones((1,) + in_shape, dtype=np.float32)
        y = self.forward(torch.tensor(x))
        self.out_shape = y.shape[1:]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        for resblock in self.resblocks:
            x = resblock(x)
        return x


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        filters: int,
        activation,
        kernel_size=(3, 3),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.conv1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(filters)
        self.act1 = activation(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(filters)
        self.act2 = activation(inplace=True)

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
    m = AlphaZeroImageBlock((96, 72, 3), n_blocks=3)
    print(m)
