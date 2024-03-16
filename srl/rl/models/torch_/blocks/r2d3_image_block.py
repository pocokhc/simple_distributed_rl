from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from srl.rl.models.torch_.converter import convert_activation_torch


class R2D3ImageBlock(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, ...],
        filters: int = 16,
        activation="ReLU",
        enable_time_distributed_layer: bool = False,
    ):
        super().__init__()
        self.enable_time_distributed_layer = enable_time_distributed_layer

        activation = convert_activation_torch(activation)

        in_ch = in_shape[-3]
        self.res1 = _ResBlock(in_ch, filters, activation)
        self.res2 = _ResBlock(filters, filters * 2, activation)
        self.res3 = _ResBlock(filters * 2, filters * 2, activation)
        self.act = activation(inplace=True)

        # --- out shape
        x = np.ones((1,) + in_shape, dtype=np.float32)
        y = self.forward(torch.tensor(x))
        self.out_shape = y.shape[-3:]

    def forward(self, x):
        if self.enable_time_distributed_layer:
            # (batch, seq, c, h, w) -> (batch*seq, c, h, w)
            batch_size, seq, channels, height, width = x.size()
            x = x.reshape((batch_size * seq, channels, height, width))

            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = self.act(x)

            # (batch*seq, c, h, w) -> (batch, seq, c, h, w)
            _, channels, height, width = x.size()
            x = x.view(batch_size, seq, channels, height, width)

        else:
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = self.act(x)

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
        self.act1 = activation(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.act2 = activation(inplace=True)
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
