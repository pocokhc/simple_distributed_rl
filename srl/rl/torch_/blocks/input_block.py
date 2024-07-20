import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from srl.base.define import SpaceTypes
from srl.base.exception import UndefinedError
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.models.config.input_config import InputImageBlockConfig, RLConfigComponentInput

logger = logging.getLogger(__name__)


def create_block_from_config(
    config: RLConfigComponentInput,
    in_space: SpaceBase,
    image_flatten: bool,
    out_multi: bool,
    dtype=np.float32,
):
    if isinstance(in_space, MultiSpace):
        return InputMultiBlock(config, in_space, concat=not out_multi, dtype=dtype)
    else:
        return InputSingleBlock(config, in_space, image_flatten, out_multi=out_multi, dtype=dtype)


class InputImageReshapeBlock(nn.Module):
    def __init__(self, space: BoxSpace, dtype):
        super().__init__()
        self.space = space

        x = np.zeros((1,) + space.shape, dtype=dtype)
        y = self.forward(torch.tensor(x))
        self.out_shape = y.shape[1:]

    def forward(self, x: torch.Tensor):
        err_msg = f"unknown space_type: {self.space}"

        if self.space.stype == SpaceTypes.GRAY_2ch:
            if len(self.space.shape) == 2:
                # (batch, h, w) -> (batch, h, w, 1)
                # (batch, h, w, 1) -> (batch, 1, h, w)
                x = x.reshape(x.shape + (1,))
                x = x.permute((0, 3, 1, 2))
            elif len(self.space.shape) == 3:
                # (batch, len, h, w)
                pass
            else:
                raise ValueError(err_msg)

        elif self.space.stype == SpaceTypes.GRAY_3ch:
            assert self.space.shape[-1] == 1
            if len(self.space.shape) == 3:
                # (batch, h, w, 1) -> (batch, 1, h, w)
                x = x.permute((0, 3, 1, 2))
            elif len(self.space.shape) == 4:
                # (batch, len, h, w, 1) -> (batch, len, h, w)
                x = x.reshape(x.shape[:4])
            else:
                raise ValueError(err_msg)

        elif self.space.stype == SpaceTypes.COLOR:
            if len(self.space.shape) == 3:
                # (batch, h, w, ch) -> (batch, ch, h, w)
                x = x.permute((0, 3, 1, 2))
            else:
                raise ValueError(err_msg)

        elif self.space.stype == SpaceTypes.IMAGE:
            if len(self.space.shape) == 3:
                # (batch, h, w, ch) -> (batch, ch, h, w)
                x = x.permute((0, 3, 1, 2))
            else:
                raise ValueError(err_msg)

        else:
            raise ValueError(err_msg)

        return x


def create_image_block_from_config(config: InputImageBlockConfig, space: SpaceBase, in_shape):
    if config._name == "DQN":
        from srl.rl.torch_.blocks.dqn_image_block import DQNImageBlock

        return DQNImageBlock(in_shape, **config._kwargs)
    if config._name == "R2D3":
        from srl.rl.torch_.blocks.r2d3_image_block import R2D3ImageBlock

        return R2D3ImageBlock(in_shape, **config._kwargs)

    if config._name == "AlphaZero":
        from srl.rl.torch_.blocks.alphazero_image_block import AlphaZeroImageBlock

        return AlphaZeroImageBlock(in_shape, **config._kwargs)
    if config._name == "MuzeroAtari":
        from srl.rl.torch_.blocks.muzero_atari_block import MuZeroAtariBlock

        return MuZeroAtariBlock(in_shape, **config._kwargs)

    if config._name == "custom":
        from srl.utils.common import load_module

        return load_module(config._kwargs["entry_point"])(in_shape, **config._kwargs["kwargs"])

    raise UndefinedError(config._name)


class InputSingleBlock(nn.Module):
    def __init__(
        self,
        config: RLConfigComponentInput,
        space: SpaceBase,
        image_flatten: bool,
        out_multi: bool,
        dtype,
    ):
        super().__init__()
        self.image_flatten = image_flatten
        self.out_multi = out_multi
        self.in_shape = space.np_shape

        self.is_image = space.is_image()
        if self.is_image:
            # --- image
            assert isinstance(space, BoxSpace)
            self.reshape_block = InputImageReshapeBlock(space, dtype)
            self.image_block = create_image_block_from_config(
                config.input_image_block,
                space,
                self.reshape_block.out_shape,
            )

            # --- flatten
            if self.image_flatten:
                self.img_flat = nn.Flatten()
                self.out_size = np.zeros(self.image_block.out_shape).flatten().shape[0]
            else:
                self.out_shape = self.image_block.out_shape
        else:
            # -- value
            self.in_flat = nn.Flatten()
            in_size = np.zeros(self.in_shape).flatten().shape[0]
            self.value_block = config.input_value_block.create_block_torch(in_size)
            self.out_size = self.value_block.out_size

    def forward(self, x: torch.Tensor):
        if self.is_image:
            x = self.reshape_block(x)
            x = self.image_block(x)
            if self.image_flatten:
                x = self.img_flat(x)
        else:
            x = self.in_flat(x)
            x = self.value_block(x)
        if self.out_multi:
            return [x]
        else:
            return x

    def create_batch_shape(self, prefix_shape: Tuple[Optional[int], ...]):
        return prefix_shape + self.in_shape

    def create_batch_single_data(self, data: np.ndarray, device):
        return torch.tensor(data[np.newaxis, ...]).to(device)

    def create_batch_stack_data(self, data: np.ndarray, device):
        # [batch_list, shape], stackはnpが早い
        return torch.tensor(np.asarray(data)).to(device)


class InputMultiBlock(nn.Module):
    def __init__(
        self,
        config: RLConfigComponentInput,
        multi_space: MultiSpace,
        concat: bool,
        dtype,
    ):
        super().__init__()
        self.concat = concat

        self.in_indices = []
        self.in_layers = nn.ModuleList()
        self.in_shapes = []
        self.out_types = []
        if concat:
            self.out_size = 0
        else:
            self.out_shapes = []
        for i, space in enumerate(multi_space.spaces):
            if not isinstance(space, BoxSpace):
                continue
            d = InputSingleBlock(
                config,
                space,
                image_flatten=concat,
                out_multi=False,
                dtype=dtype,
            )
            self.in_indices.append(i)
            self.in_layers.append(d)
            self.in_shapes.append(space.shape)
            self.out_types.append(space.stype)
            if concat:
                self.out_size += d.out_size
            else:
                self.out_shapes.append(d.out_shape)
        assert len(self.in_indices) > 0

    def forward(self, x: torch.Tensor):
        x_arr = []
        i = -1
        for idx in self.in_indices:
            i += 1
            _x = x[idx]
            _x = self.in_layers[i](_x)
            x_arr.append(_x)
        if self.concat:
            x_arr = torch.cat(x_arr, dim=-1)
        return x_arr

    def create_batch_shape(self, prefix_shape: Tuple[Optional[int], ...]):
        return [prefix_shape + s for s in self.in_shapes]

    def create_batch_single_data(self, data: List[np.ndarray], device):
        return [torch.tensor(d[np.newaxis, ...]).to(device) for d in data]

    def create_batch_stack_data(self, data: np.ndarray, device):
        # [batch_list, multi_list, shape], stackはnpが早い
        arr = []
        for idx in self.in_indices:
            arr.append(torch.tensor(np.asarray([d[idx] for d in data])).to(device))
        return arr
