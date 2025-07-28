import numpy as np
import torch
import torch.nn as nn

from srl.base.spaces.box import BoxSpace
from srl.rl.torch_.blocks.input_image_block import InputImageReshapeBlock

from .config import Config


class ResBlock(nn.Module):
    def __init__(self, channels: int, output_scale_factor: float = 2):
        super().__init__()
        self.output_scale_factor = output_scale_factor

        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        x1 = self.act1(x)
        x1 = self.conv1(x1)
        x1 = self.act2(x1)
        x1 = self.conv2(x1)
        return (x + x1) / self.output_scale_factor


def DQNImageEncoder(units: int, out_units: int, space: BoxSpace):
    reshape_block = InputImageReshapeBlock(space)
    in_ch = reshape_block.out_shape[-3]

    return nn.Sequential(
        reshape_block,
        nn.Conv2d(in_ch, units, kernel_size=8, stride=4, padding=3, padding_mode="replicate"),
        nn.SiLU(),
        nn.Conv2d(units, units * 2, kernel_size=4, stride=2, padding=2, padding_mode="replicate"),
        nn.SiLU(),
        nn.Conv2d(units * 2, units * 2, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
        nn.SiLU(),
        nn.Flatten(),
        nn.LazyLinear(out_units),
        nn.LayerNorm(out_units),
        nn.SiLU(),
    )


def R2D3ImageEncoder(units: int, out_units: int, space):
    reshape_block = InputImageReshapeBlock(space)
    in_ch = reshape_block.out_shape[-3]
    half_units = units // 2
    return nn.Sequential(
        reshape_block,
        # 1
        nn.Conv2d(in_ch, half_units, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        ResBlock(half_units),
        ResBlock(half_units),
        # 2
        nn.Conv2d(half_units, units, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        ResBlock(units),
        ResBlock(units),
        # 3
        nn.Conv2d(units, units, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        ResBlock(units),
        ResBlock(units),
        nn.ReLU(),
        # out
        nn.Flatten(),
        nn.LazyLinear(out_units),
        nn.LayerNorm(out_units),
        nn.SiLU(),
    )


class DiscreteEncoder(nn.Module):
    def __init__(self, units: int, out_units: int, space: BoxSpace):
        super().__init__()

        low = torch.tensor(space.low, dtype=torch.long)
        self.register_buffer("low", low.unsqueeze(0))

        emb_size = int(np.max(space.high - space.low))
        self.layers = nn.ModuleList(
            [
                nn.Embedding(emb_size, units),
                nn.Flatten(),
                nn.LazyLinear(out_units),
                nn.LayerNorm(out_units),
                nn.SiLU(),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long() - self.low  # type: ignore
        for layer in self.layers:
            x = layer(x)
        return x


def ContEncoder(units: int, out_units: int):
    return nn.Sequential(
        nn.Flatten(),
        nn.LazyLinear(units),
        nn.LayerNorm(units),
        nn.SiLU(),
        nn.Linear(units, out_units),
        nn.SiLU(),
    )


# --------------------------------------


def create_encoder_block(config: Config):
    units = config.base_units
    space = config.observation_space

    if space.is_image():
        out_units = min(config.base_units, 256)
        if config.encode_img_type == "DQN":
            units = 32
            obs_block = DQNImageEncoder(units, out_units, space)
        elif config.encode_img_type == "R2D3":
            units = 16
            obs_block = R2D3ImageEncoder(units, out_units, space)
    elif config.used_discrete_block and space.is_discrete():
        out_units = min(config.base_units, 128)

        # embの出力units数を制限
        max_units = 8192
        emb_size = int(np.max(space.high - space.low)) * space.flatten_size
        emb_units = max_units // emb_size  # max_units以下になるように計算
        # 下限以下の場合はcontにする
        if emb_units < 16:
            out_units = min(config.base_units, 128)
            obs_block = ContEncoder(units, out_units)
        else:
            emb_units = min(emb_units, units)  # 上限はunits数
            obs_block = DiscreteEncoder(emb_units, out_units, space)
    else:
        out_units = min(config.base_units, 128)
        obs_block = ContEncoder(units, out_units)

    return obs_block
