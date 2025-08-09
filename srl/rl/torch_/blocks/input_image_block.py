import logging

import numpy as np
import torch
import torch.nn as nn

from srl.base.exception import UndefinedError
from srl.base.spaces.box import BoxSpace
from srl.rl.models.config.input_block import InputImageBlockConfig
from srl.rl.torch_.blocks.input_image_reshape_block import InputImageReshapeBlock

logger = logging.getLogger(__name__)


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


class InputImageBlock(nn.Module):
    def __init__(
        self,
        config: InputImageBlockConfig,
        in_space: BoxSpace,
        out_flatten: bool,
        reshape_for_rnn: bool,
    ):
        super().__init__()
        self.in_shape = in_space.shape
        self.out_flatten = out_flatten
        self.reshape_for_rnn = reshape_for_rnn

        self.reshape_block = InputImageReshapeBlock(in_space)
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
