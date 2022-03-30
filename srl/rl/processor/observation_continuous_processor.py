from typing import Tuple

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType, RLObservationType
from srl.base.rl.processor import Processor


class ObservationContinuousProcessor(Processor):
    def change_observation_info(
        self,
        observation_space: gym.spaces.Box,
        observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
    ) -> Tuple[gym.spaces.Box, EnvObservationType]:
        if rl_observation_type != RLObservationType.CONTINUOUS:
            return observation_space, observation_type

        return observation_space, observation_type

    def observation_encode(self, observation: np.ndarray) -> np.ndarray:
        return observation


if __name__ == "__main__":
    pass
