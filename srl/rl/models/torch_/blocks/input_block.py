import logging
from typing import List, cast

import numpy as np
import torch
import torch.nn as nn

from srl.base.define import EnvTypes, RLTypes
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.models.image_block import ImageBlockConfig
from srl.rl.models.mlp_block import MLPBlockConfig

logger = logging.getLogger(__name__)


class _InputImageBlock(nn.Module):
    def __init__(self, obs_space: SpaceBase):
        super().__init__()
        self.obs_space = obs_space

        x = np.zeros((1,) + obs_space.shape, dtype=np.float32)
        y = self.forward(torch.tensor(x))
        self.out_shape = y.shape[1:]

    def forward(self, x: torch.Tensor):
        err_msg = f"unknown observation_type: {self.obs_space}"

        if self.obs_space.env_type == EnvTypes.GRAY_2ch:
            if len(self.obs_space.shape) == 2:
                # (batch, h, w) -> (batch, h, w, 1)
                # (batch, h, w, 1) -> (batch, 1, h, w)
                x = x.reshape(x.shape + (1,))
                x = x.permute((0, 3, 1, 2))
            elif len(self.obs_space.shape) == 3:
                # (batch, len, h, w)
                pass
            else:
                raise ValueError(err_msg)

        elif self.obs_space.env_type == EnvTypes.GRAY_3ch:
            assert self.obs_space.shape[-1] == 1
            if len(self.obs_space.shape) == 3:
                # (batch, h, w, 1) -> (batch, 1, h, w)
                x = x.permute((0, 3, 1, 2))
            elif len(self.obs_space.shape) == 4:
                # (batch, len, h, w, 1) -> (batch, len, h, w)
                x = x.reshape(x.shape[:4])
            else:
                raise ValueError(err_msg)

        elif self.obs_space.env_type == EnvTypes.COLOR:
            if len(self.obs_space.shape) == 3:
                # (batch, h, w, ch) -> (batch, ch, h, w)
                x = x.permute((0, 3, 1, 2))
            else:
                raise ValueError(err_msg)

        elif self.obs_space.env_type == EnvTypes.IMAGE:
            if len(self.obs_space.shape) == 3:
                # (batch, h, w, ch) -> (batch, ch, h, w)
                x = x.permute((0, 3, 1, 2))
            else:
                raise ValueError(err_msg)

        else:
            raise ValueError(err_msg)

        return x


class InputBlock(nn.Module):
    def __init__(
        self,
        value_block_config: MLPBlockConfig,
        image_block_config: ImageBlockConfig,
        observation_space: SpaceBase,
        enable_time_distributed_layer: bool = False,
    ):
        super().__init__()
        self.enable_time_distributed_layer = enable_time_distributed_layer

        if isinstance(observation_space, MultiSpace):
            self.is_multi_input = True
            self.in_multi_layers = nn.ModuleList()
            self.in_multi_size = len(observation_space.spaces)
            self.out_size = 0
            for space in observation_space.spaces:
                layers = nn.ModuleList()
                if space.rl_type == RLTypes.IMAGE:
                    in_img_block = _InputImageBlock(space)
                    layers.append(in_img_block)
                    img_block = image_block_config.create_block_torch(in_img_block.out_shape)
                    layers.append(img_block)
                    layers.append(nn.Flatten())
                    self.out_size += np.zeros(img_block.out_shape).flatten().shape[0]
                    self.in_multi_layers.append(layers)
                else:
                    layers.append(nn.Flatten())
                    flat_size = np.zeros(space.shape).flatten().shape[0]
                    val_block = value_block_config.create_block_torch(flat_size)
                    layers.append(val_block)
                    self.out_size += val_block.out_size
                    self.in_multi_layers.append(layers)

        else:
            self.is_multi_input = False
            self.in_layers = nn.ModuleList()
            if observation_space.rl_type == RLTypes.IMAGE:
                in_img_block = _InputImageBlock(observation_space)
                self.in_layers.append(in_img_block)
                img_block = image_block_config.create_block_torch(in_img_block.out_shape)
                self.in_layers.append(img_block)
                self.in_layers.append(nn.Flatten())
                self.out_size = np.zeros(img_block.out_shape).flatten().shape[0]
            else:
                self.in_layers.append(nn.Flatten())
                flat_size = np.zeros(observation_space.shape).flatten().shape[0]
                val_block = value_block_config.create_block_torch(flat_size)
                self.in_layers.append(val_block)
                self.out_size = val_block.out_size

    def forward(self, x: torch.Tensor):
        if self.is_multi_input:
            _x = []
            for i in range(self.in_multi_size):
                _x2 = x[i]
                _x2, batch_size, seq_size = self._sequence_to_batch(_x2)
                for h in self.in_multi_layers[i]:
                    _x2 = h(_x2)
                _x2 = self._batch_to_sequence(_x2, batch_size, seq_size)
                _x.append(_x2)
            x = torch.cat(_x, dim=-1)
        else:
            x, batch_size, seq_size = self._sequence_to_batch(x)
            for h in self.in_layers:
                x = h(x)
            x = self._batch_to_sequence(x, batch_size, seq_size)

        return x

    def _sequence_to_batch(self, x: torch.Tensor):
        batch_size = 0
        seq_size = 0
        if self.enable_time_distributed_layer:
            # (batch, seq, shape) -> (batch*seq, shape)
            size = x.size()
            batch_size = size[0]
            seq_size = size[1]
            shape = size[2:]
            x = x.reshape((batch_size * seq_size, *shape))
        return x, batch_size, seq_size

    def _batch_to_sequence(self, x: torch.Tensor, batch_size: int, seq_size: int):
        if self.enable_time_distributed_layer:
            # (batch*seq, shape) -> (batch, seq, shape)
            size = x.size()
            shape = size[1:]
            x = x.view(batch_size, seq_size, *shape)
        return x
