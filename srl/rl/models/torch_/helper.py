from typing import cast

import numpy as np
import torch

from srl.base.define import RLTypes
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase


def create_batch_data(state, space: SpaceBase, device):
    if space.rl_type == RLTypes.MULTI:
        return [torch.tensor(s[np.newaxis, ...]).to(device) for s in state]
    else:
        return torch.tensor(state[np.newaxis, ...]).to(device)


def stack_batch_data(state, space: SpaceBase, device):
    # stackはnpが早い

    if space.rl_type == RLTypes.MULTI:
        space = cast(MultiSpace, space)
        # state: [batch_list, multi_list, state_shape]
        states = []
        for i in range(space.space_size):
            states.append(torch.tensor(np.asarray([s[i] for s in state])).to(device))
        return states
    else:
        # state: [batch_list, state_shape]
        return torch.tensor(np.asarray(state)).to(device)


def encode_sequence_batch(x: torch.Tensor):
    # (batch, seq, shape) -> (batch*seq, shape)
    size = x.size()
    head_size1 = size[0]
    head_size2 = size[1]
    shape = size[2:]
    x = x.reshape((head_size1 * head_size2, *shape))
    return x, head_size1, head_size2


def decode_sequence_batch(x: torch.Tensor, head_size1: int, head_size2: int):
    # (batch*seq, shape) -> (batch, seq, shape)
    size = x.size()
    shape = size[1:]
    x = x.view(head_size1, head_size2, *shape)
    return x
