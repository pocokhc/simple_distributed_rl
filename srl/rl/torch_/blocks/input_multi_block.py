import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn

from srl.base.spaces.box import BoxSpace
from srl.base.spaces.multi import MultiSpace
from srl.rl.models.config.input_image_block import InputImageBlockConfig
from srl.rl.models.config.input_value_block import InputValueBlockConfig
from srl.rl.torch_.blocks.a_input_block import AInputBlock

logger = logging.getLogger(__name__)


class InputMultiBlockConcat(nn.Module, AInputBlock):
    def __init__(
        self,
        multi_space: MultiSpace,
        value_block_config: InputValueBlockConfig,
        image_block_config: InputImageBlockConfig,
        reshape_for_rnn: List[bool],
    ):
        super().__init__()

        self.in_indices = []
        self.in_blocks = nn.ModuleList()
        self.in_shapes = []
        self.out_types = []

        self.out_size = 0
        for i, space in enumerate(multi_space.spaces):
            if not isinstance(space, BoxSpace):
                continue
            if space.is_value():
                d = value_block_config.create_torch_block(
                    space.shape,
                    input_flatten=True,
                    reshape_for_rnn=reshape_for_rnn[i],
                )
                self.in_blocks.append(d)
            elif space.is_image():
                d = image_block_config.create_torch_block(
                    space,
                    out_flatten=True,
                    reshape_for_rnn=reshape_for_rnn[i],
                )
                self.in_blocks.append(d)
            else:
                continue
            self.in_indices.append(i)
            self.in_shapes.append(space.shape)
            self.out_types.append(space.stype)
            self.out_size += d.out_size
        assert len(self.in_indices) > 0

    def forward(self, x: torch.Tensor):
        x_arr = []
        i = -1
        for idx in self.in_indices:
            i += 1
            _x = x[idx]
            _x = self.in_blocks[i](_x)
            x_arr.append(_x)
        x_arr = torch.cat(x_arr, dim=-1)
        return x_arr

    # -----------------------

    def to_torch_one_batch(self, data, device, torch_dtype, add_expand_dim: bool = True):
        if add_expand_dim:
            return [torch.tensor(data[i], dtype=torch_dtype, device=device).unsqueeze(0) for i in self.in_indices]
        else:
            return [torch.tensor(data[i], dtype=torch_dtype, device=device) for i in self.in_indices]

    def to_torch_batches(self, data, device, torch_dtype):
        return [torch.tensor(np.asarray([d[i] for d in data], dtype=torch_dtype, device=device)) for i in self.in_indices]
