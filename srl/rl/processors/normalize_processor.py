from dataclasses import dataclass
from typing import List, Optional, Tuple, cast

import numpy as np

from srl.base.define import EnvObservationType
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.continuous import ContinuousSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.np_array import NpArraySpace
from srl.base.spaces.space import SpaceBase


@dataclass
class NormalizeProcessor(RLProcessor):
    feature_range: Tuple[float, float] = (-1, 1)

    def remap_observation_space(self, prev_space: SpaceBase, **kwargs) -> Optional[SpaceBase]:
        assert self.feature_range[0] <= self.feature_range[1]

        if isinstance(prev_space, DiscreteSpace):
            return ContinuousSpace(self.feature_range[0], self.feature_range[1])

        if isinstance(prev_space, ContinuousSpace):
            return ContinuousSpace(self.feature_range[0], self.feature_range[1])

        if isinstance(prev_space, ArrayDiscreteSpace):
            return NpArraySpace(prev_space._size, self.feature_range[0], self.feature_range[1])

        if isinstance(prev_space, NpArraySpace):
            return NpArraySpace(prev_space._size, self.feature_range[0], self.feature_range[1])

        if isinstance(prev_space, BoxSpace):
            return BoxSpace(prev_space.shape, self.feature_range[0], self.feature_range[1])

        return None

    def remap_observation(self, state: EnvObservationType, prev_space: SpaceBase, new_space: SpaceBase, **kwargs) -> EnvObservationType:
        _min = self.feature_range[0]
        _max = self.feature_range[1]

        if isinstance(prev_space, DiscreteSpace):
            state = cast(int, state)
            state = state / (prev_space.n - 1)
            return float(state * (_max - _min) + _min)

        if isinstance(prev_space, ContinuousSpace):
            state = cast(float, state)
            _low = prev_space.low
            _high = prev_space.high
            state = ((state - _low) / (_high - _low)) * (_max - _min) + _min
            return state

        if isinstance(prev_space, ArrayDiscreteSpace):
            state = cast(List[int], state)
            state = state[:]  # copy
            for i in range(prev_space.size):
                _low = prev_space.low[i]
                _high = prev_space.high[i]
                state[i] = ((state[i] - _low) / (_high - _low)) * (_max - _min) + _min
            return state

        if isinstance(prev_space, NpArraySpace):
            state = cast(List[float], state)
            state = state[:]  # copy
            for i in range(prev_space.size):
                _low = prev_space.low[i]
                _high = prev_space.high[i]
                state[i] = ((state[i] - _low) / (_high - _low)) * (_max - _min) + _min
            return state

        if isinstance(prev_space, BoxSpace):
            state = cast(np.ndarray, state)
            _low = prev_space.low
            _high = prev_space.high
            state = ((state - _low) / (_high - _low)) * (_max - _min) + _min
            return state

        return state
