from typing import cast

import numpy as np

from srl.base.define import RLTypes
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase


def create_batch_data(state, space: SpaceBase):
    if space.rl_type == RLTypes.MULTI:
        return [s[np.newaxis, ...] for s in state]
    else:
        return state[np.newaxis, ...]


def stack_batch_data(state, space: SpaceBase):
    # stackはnpが早い

    if space.rl_type == RLTypes.MULTI:
        space = cast(MultiSpace, space)
        # state: [batch_list, multi_list, state_shape]
        states = []
        for i in range(space.space_size):
            states.append(np.asarray([s[i] for s in state]))
        return states
    else:
        # state: [batch_list, state_shape]
        return np.asarray(state)
