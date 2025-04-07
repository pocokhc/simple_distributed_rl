import logging

import numpy as np
import torch
import torch.nn as nn

from srl.base.define import SpaceTypes
from srl.base.exception import UndefinedError
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.models.config.input_image_block import InputImageBlockConfig
from srl.rl.torch_.blocks.a_input_block import AInputBlock

logger = logging.getLogger(__name__)


class InputImageBlock(nn.Module, AInputBlock):
    def __init__(
        self,
        config: InputImageBlockConfig,
        space: SpaceBase,
        out_flatten: bool,
        reshape_for_rnn: bool,
    ):
        super().__init__()
        self.in_shape = space.np_shape
        self.out_flatten = out_flatten
        self.reshape_for_rnn = reshape_for_rnn

        assert space.is_image(), f"Only image space can be used. {space=}"
        assert isinstance(space, BoxSpace)
        self.reshape_block = InputImageReshapeBlock(space)
        self.image_block = create_block_from_config(config, self.reshape_block.out_shape)

        if self.out_flatten:
            self.img_flat = nn.Flatten()
            self.out_size = np.zeros(self.image_block.out_shape).flatten().shape[0]
        else:
            self.out_shape = self.image_block.out_shape

    def forward(self, x: torch.Tensor):
        if self.reshape_for_rnn:
            size = x.size()
            self.batch_size = size[0]
            self.seq_len = size[1]
            feat_dims = size[2:]
            x = x.reshape((self.batch_size * self.seq_len, *feat_dims))
        x = self.reshape_block(x)
        x = self.image_block(x)
        if self.out_flatten:
            x = self.img_flat(x)
        return x

    def unreshape_for_rnn(self, x):
        return x.view(self.batch_size, self.seq_len, *x.size()[1:])


def create_block_from_config(config: InputImageBlockConfig, in_shape):
    if config.name == "DQN":
        from srl.rl.torch_.blocks.dqn_image_block import DQNImageBlock

        return DQNImageBlock(in_shape, **config.kwargs)
    if config.name == "R2D3":
        from srl.rl.torch_.blocks.r2d3_image_block import R2D3ImageBlock

        return R2D3ImageBlock(in_shape, **config.kwargs)

    if config.name == "AlphaZero":
        from srl.rl.torch_.blocks.alphazero_image_block import AlphaZeroImageBlock

        return AlphaZeroImageBlock(in_shape, **config.kwargs)
    if config.name == "MuzeroAtari":
        from srl.rl.torch_.blocks.muzero_atari_block import MuZeroAtariBlock

        return MuZeroAtariBlock(in_shape, **config.kwargs)

    if config.name == "custom":
        from srl.utils.common import load_module

        return load_module(config.kwargs["entry_point"])(in_shape, **config.kwargs["kwargs"])

    raise UndefinedError(config.name)


class InputImageReshapeBlock(nn.Module):
    def __init__(self, space: BoxSpace):
        super().__init__()
        self.space = space

        x = np.zeros((1,) + space.shape)
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
