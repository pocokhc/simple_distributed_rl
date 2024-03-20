from typing import Optional, Tuple, cast

import numpy as np

from srl.base.define import SpaceTypes
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase


def create_batch_shape(shape, prefix_shape: Tuple[Optional[int], ...]):
    if isinstance(shape, list):
        return [prefix_shape + s for s in shape]
    else:
        return prefix_shape + shape


def create_batch_data(state, space: SpaceBase):
    if space.stype == SpaceTypes.MULTI:
        return [s[np.newaxis, ...] for s in state]
    else:
        return state[np.newaxis, ...]


def stack_batch_data(state, space: SpaceBase):
    # stackはnpが早い

    if space.stype == SpaceTypes.MULTI:
        space = cast(MultiSpace, space)
        # state: [batch_list, multi_list, state_shape]
        states = []
        for i in range(space.space_size):
            states.append(np.asarray([s[i] for s in state]))
        return states
    else:
        # state: [batch_list, state_shape]
        return np.asarray(state)
