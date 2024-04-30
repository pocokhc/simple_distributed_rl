from dataclasses import dataclass
from typing import List, Tuple, cast

import numpy as np

from srl.base.define import EnvObservationType
from srl.base.env.env_run import EnvRun, SpaceBase
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.rl.worker_run import WorkerRun
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.continuous import ContinuousSpace
from srl.base.spaces.discrete import DiscreteSpace


@dataclass
class NormalizeProcessor(Processor):
    feature_rang: Tuple[float, float] = (0, 1)

    def remap_observation_space(self, env_observation_space: SpaceBase, env: EnvRun, rl_config: RLConfig) -> SpaceBase:
        self._old_space = env_observation_space
        assert self.feature_rang[0] <= self.feature_rang[1]

        if isinstance(env_observation_space, DiscreteSpace):
            return ContinuousSpace(self.feature_rang[0], self.feature_rang[1])

        if isinstance(env_observation_space, ContinuousSpace):
            return ContinuousSpace(self.feature_rang[0], self.feature_rang[1])

        if isinstance(env_observation_space, ArrayDiscreteSpace):
            return ArrayContinuousSpace(env_observation_space._size, self.feature_rang[0], self.feature_rang[1])

        if isinstance(env_observation_space, ArrayContinuousSpace):
            return ArrayContinuousSpace(env_observation_space._size, self.feature_rang[0], self.feature_rang[1])

        if isinstance(env_observation_space, BoxSpace):
            return BoxSpace(env_observation_space.shape, self.feature_rang[0], self.feature_rang[1])

        return env_observation_space

    def remap_observation(self, state: EnvObservationType, worker: WorkerRun, env: EnvRun) -> EnvObservationType:
        _min = self.feature_rang[0]
        _max = self.feature_rang[1]

        if isinstance(self._old_space, DiscreteSpace):
            state = cast(int, state)
            state = state / (self._old_space.n - 1)
            return float(state * (_max - _min) + _min)

        if isinstance(self._old_space, ContinuousSpace):
            state = cast(float, state)
            _low = self._old_space.low
            _high = self._old_space.high
            state = ((state - _low) / (_high - _low)) * (_max - _min) + _min
            return state

        if isinstance(self._old_space, ArrayDiscreteSpace):
            state = cast(List[int], state)
            state = state[:]  # copy
            for i in range(self._old_space.size):
                _low = self._old_space.low[i]
                _high = self._old_space.high[i]
                state[i] = ((state[i] - _low) / (_high - _low)) * (_max - _min) + _min
            return state

        if isinstance(self._old_space, ArrayContinuousSpace):
            state = cast(List[float], state)
            state = state[:]  # copy
            for i in range(self._old_space.size):
                _low = self._old_space.low[i]
                _high = self._old_space.high[i]
                state[i] = ((state[i] - _low) / (_high - _low)) * (_max - _min) + _min
            return state

        if isinstance(self._old_space, BoxSpace):
            state = cast(np.ndarray, state)
            _low = self._old_space.low
            _high = self._old_space.high
            state = ((state - _low) / (_high - _low)) * (_max - _min) + _min
            return state

        return state
