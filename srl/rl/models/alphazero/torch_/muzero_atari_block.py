from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class MuZeroAtariBlock(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        filters: int = 128,
        use_layer_normalization: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        in_ch = in_shape[0]
        self.conv1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=filters,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            bias=False,
        )
        self.act1 = nn.ReLU(inplace=True)
        self.resblock1 = _ResidualBlock(filters, use_layer_normalization)
        self.resblock2 = _ResidualBlock(filters, use_layer_normalization)
        self.conv2 = nn.Conv2d(
            in_channels=filters,
            out_channels=filters * 2,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            bias=False,
        )
        self.act2 = nn.ReLU(inplace=True)
        self.resblock3 = _ResidualBlock(filters * 2, use_layer_normalization)
        self.resblock4 = _ResidualBlock(filters * 2, use_layer_normalization)
        self.resblock5 = _ResidualBlock(filters * 2, use_layer_normalization)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblock6 = _ResidualBlock(filters * 2, use_layer_normalization)
        self.resblock7 = _ResidualBlock(filters * 2, use_layer_normalization)
        self.resblock8 = _ResidualBlock(filters * 2, use_layer_normalization)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # --- out shape
        x = np.ones((1,) + in_shape, dtype=np.float32)
        y = self.forward(torch.tensor(x))
        self.out_shape = y.shape[1:]

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.pool1(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.pool2(x)
        return x


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        filters: int,
        use_layer_normalization: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

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
        self.act1 = nn.ReLU(inplace=True)
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
        self.act2 = nn.ReLU(inplace=True)

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
