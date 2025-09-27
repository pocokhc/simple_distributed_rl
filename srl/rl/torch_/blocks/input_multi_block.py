import math
from typing import List, cast

import numpy as np
import torch
import torch.nn as nn

from srl.base.spaces.box import BoxSpace
from srl.base.spaces.multi import MultiSpace
from srl.rl.models.config.input_multi_block import InputMultiBlockConfig


class InputMultiBlock(nn.Module):
    def __init__(self, config: InputMultiBlockConfig, in_space: MultiSpace):
        super().__init__()

        self.blocks = []
        self.out_size = 0
        for space in in_space.spaces:
            block, size = create_encoder_block(config, cast(BoxSpace, space))
            self.out_size += size
            self.blocks.append(block)
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x_list: List[torch.Tensor]):
        y_list = []
        for i in range(len(self.blocks)):
            y_list.append(self.blocks[i](x_list[i]))
        y = torch.cat(y_list, dim=-1)
        return y


def create_encoder_block(config: InputMultiBlockConfig, space: BoxSpace):
    if space.is_image_visual():
        if config.image_type == "DQN":
            from srl.rl.torch_.blocks.dqn_image_block import DQNImageBlock

            img_block = DQNImageBlock((space.shape[2], space.shape[0], space.shape[1]))
            obs_block = nn.Sequential(
                NHWC_to_NCHW(),
                img_block,
                nn.Flatten(),
            )
            out_size = math.prod(img_block.out_shape)
        elif config.image_type == "R2D3":
            from srl.rl.torch_.blocks.r2d3_image_block import R2D3ImageBlock

            img_block = R2D3ImageBlock((space.shape[2], space.shape[0], space.shape[1]))
            obs_block = nn.Sequential(
                NHWC_to_NCHW(),
                img_block,
                nn.Flatten(),
            )
            out_size = math.prod(img_block.out_shape)
    elif space.is_discrete():
        if config.discrete_type == "BOX":
            obs_block = ContEncoder(config.cont_units)
            out_size = config.cont_units
        else:
            # obs * enb_units = params
            target_units = config.discrete_target_params // space.flatten_size
            emb_units = min(target_units, config.discrete_units)
            emb_units = max(emb_units, config.discrete_low_units)
            if config.discrete_type == "Discrete":
                obs_block = DiscreteEncoder(emb_units, config.discrete_units, space)
            elif config.discrete_type == "Conv1D":
                obs_block = DiscreteConv1DEncoder(emb_units, config.discrete_units, space)
            out_size = config.discrete_units
    else:
        obs_block = ContEncoder(config.cont_units)
        out_size = config.cont_units

    return obs_block, out_size


class NHWC_to_NCHW(nn.Module):
    """入力を (N, H, W, C) -> (N, C, H, W) に変換するレイヤー"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2)


class DiscreteEncoder(nn.Module):
    def __init__(self, emb_units: int, out_units: int, space: BoxSpace):
        super().__init__()

        emb_size = int(np.max(space.high - space.low)) + 1
        self.block = nn.Sequential(
            nn.Flatten(),
            nn.Embedding(emb_size, emb_units),
            nn.Flatten(),
            nn.LazyLinear(out_units),
            nn.LayerNorm(out_units),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x.long())


class ResidualBlock1D(nn.Module):
    """
    1D畳み込みのResidualブロック
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, output_scale_factor: float = 2.0):
        super().__init__()
        self.output_scale_factor = output_scale_factor
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.block(x) + x) / self.output_scale_factor


class DiscreteConv1DEncoder(nn.Module):
    def __init__(self, emb_units: int, out_units: int, space: BoxSpace):
        super().__init__()

        emb_size = int(np.max(space.high - space.low)) + 1
        self.in_flatten = nn.Flatten()
        self.embedding = nn.Embedding(emb_size, emb_units)
        self.block = nn.Sequential(
            ResidualBlock1D(emb_units, emb_units),
            ResidualBlock1D(emb_units, emb_units),
            nn.Flatten(),
            nn.LazyLinear(out_units),
            nn.LayerNorm(out_units),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_flatten(x.long())
        x = self.embedding(x)
        x = x.transpose(1, 2)
        return self.block(x)


class ContEncoder(nn.Module):
    def __init__(self, units: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(units),
            nn.LayerNorm(units),
            nn.SiLU(),
            nn.Linear(units, units),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
