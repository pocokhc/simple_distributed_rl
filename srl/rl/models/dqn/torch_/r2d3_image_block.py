from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class R2D3ImageBlock(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, ...],
        enable_time_distributed_layer: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_time_distributed_layer = enable_time_distributed_layer

        in_ch = in_shape[-3]
        self.res1 = _ResBlock(in_ch, 16)
        self.res2 = _ResBlock(16, 32)
        self.res3 = _ResBlock(32, 32)
        self.relu = nn.ReLU(inplace=True)

        # --- out shape
        x = np.ones((1,) + in_shape, dtype=np.float32)
        y = self.forward(torch.tensor(x))
        self.out_shape = y.shape[-3:]

    def forward(self, x):
        if self.enable_time_distributed_layer:
            # (batch, seq, c, h, w) -> (batch*seq, c, h, w)
            batch_size, seq, channels, height, width = x.size()
            x = x.view(batch_size * seq, channels, height, width)

            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = self.relu(x)

            # (batch*seq, c, h, w) -> (batch, seq, c, h, w)
            _, channels, height, width = x.size()
            x = x.view(batch_size, seq, channels, height, width)

        else:
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = self.relu(x)

        return x


class _ResBlock(nn.Module):
    def __init__(self, in_ch, n_filter):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=n_filter,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.res1 = _ResidualBlock(n_filter)
        self.res2 = _ResidualBlock(n_filter)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class _ResidualBlock(nn.Module):
    def __init__(self, n_filter):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels=n_filter,
            out_channels=n_filter,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=n_filter,
            out_channels=n_filter,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.add = nn.Identity()

    def forward(self, x):
        x1 = self.relu1(x)
        x1 = self.conv1(x1)
        x1 = self.relu2(x1)
        x1 = self.conv2(x1)
        return self.add(x) + x1


if __name__ == "__main__":
    m = R2D3ImageBlock((96, 96, 3))
    print(m)
