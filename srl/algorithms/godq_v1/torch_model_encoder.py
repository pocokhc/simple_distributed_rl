import numpy as np
import torch
import torch.nn as nn

from srl.base.spaces.box import BoxSpace
from srl.rl.torch_.blocks.input_image_reshape_block import InputImageReshapeBlock

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

        self.register_buffer("low", torch.tensor(space.low, dtype=torch.long).unsqueeze(0))
        emb_size = int(np.max(space.high - space.low)) + 1
        emb_size = min(emb_size, units)  # 上限を設定

        self.block = nn.Sequential(
            nn.Flatten(),
            nn.Embedding(emb_size, units),
            nn.Flatten(),
            nn.LazyLinear(out_units),
            nn.LayerNorm(out_units),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long() - self.low  # type: ignore
        return self.block(x)


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
    def __init__(self, units: int, out_units: int, space: BoxSpace):
        super().__init__()

        self.register_buffer("low", torch.tensor(space.low, dtype=torch.long).unsqueeze(0))
        emb_size = int(np.max(space.high - space.low)) + 1
        emb_size = min(emb_size, units)  # 上限を設定

        self.in_flatten = nn.Flatten()
        self.embedding = nn.Embedding(emb_size, units)
        self.block = nn.Sequential(
            ResidualBlock1D(units, units),
            ResidualBlock1D(units, units),
            nn.Flatten(),
            nn.LazyLinear(out_units),
            nn.LayerNorm(out_units),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long() - self.low  # type: ignore
        x = self.in_flatten(x)
        x = self.embedding(x)
        x = x.transpose(1, 2)
        return self.block(x)


class ContEncoder(nn.Module):
    def __init__(self, units: int, out_units: int, space: BoxSpace, torch_dtype, enable_norm: bool):
        super().__init__()
        self.enable_norm = enable_norm

        self.register_buffer("low", torch.tensor(space.low, dtype=torch_dtype).unsqueeze(0))
        diff = torch.tensor(2 / (space.high - space.low), dtype=torch_dtype)
        self.register_buffer("diff", diff.unsqueeze(0))
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(units),
            nn.LayerNorm(units),
            nn.SiLU(),
            nn.Linear(units, out_units),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_norm:
            x = (x - self.low) * self.diff - 1  # type: ignore
        return self.layers(x)


# --------------------------------------


def create_encoder_block(config: Config):
    base_units = config.base_units
    space = config.observation_space

    out_units = base_units
    if space.is_image():
        if config.encode_img_type == "DQN":
            units = 32
            obs_block = DQNImageEncoder(units, out_units, space)
        elif config.encode_img_type == "R2D3":
            units = 16
            obs_block = R2D3ImageEncoder(units, out_units, space)
    elif config.used_discrete_block and space.is_discrete():
        if config.encode_discrete_type == "BOX":
            obs_block = ContEncoder(base_units, out_units, space, config.get_dtype("torch"), config.enable_state_norm)
        elif config.encode_discrete_type == "Discrete":
            obs_block = DiscreteEncoder(base_units, out_units, space)
        elif config.encode_discrete_type == "Conv1D":
            obs_block = DiscreteConv1DEncoder(base_units, out_units, space)
    else:
        obs_block = ContEncoder(base_units, out_units, space, config.get_dtype("torch"), config.enable_state_norm)

    # --- init weight
    x = torch.zeros((1, *config.observation_space.shape), dtype=config.get_dtype("torch"))
    obs_block(x)

    return obs_block, out_units
