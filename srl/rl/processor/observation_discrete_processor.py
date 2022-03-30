import logging
from dataclasses import dataclass
from typing import Any, Tuple

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType, RLObservationType
from srl.base.rl.processor import Processor


@dataclass
class ObservationDiscreteProcessor(Processor):

    observation_division_num: int = 50

    def __post_init__(self):
        self._observation_discrete_diff = None
        self.low = None

    def change_observation_info(
        self,
        observation_space: gym.spaces.Box,
        observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
    ) -> Tuple[gym.spaces.Box, EnvObservationType]:
        if rl_observation_type != RLObservationType.DISCRETE:
            return observation_space, observation_type

        self._observation_discrete_diff = None

        if observation_type != EnvObservationType.CONTINUOUS:
            return observation_space, observation_type

        # 離散化
        division_num = self.observation_division_num
        self.low = observation_space.low
        high = observation_space.high
        self._observation_discrete_diff = (high - self.low) / division_num

        observation_type = EnvObservationType.DISCRETE
        return observation_space, observation_type

    def observation_encode(self, observation: np.ndarray) -> np.ndarray:
        if self._observation_discrete_diff is None:
            return observation

        next_state = (observation - self.low) / self._observation_discrete_diff
        return np.int64(next_state)


if __name__ == "__main__":
    pass
