from abc import ABC
from typing import List, Tuple

import numpy as np
from srl.base.define import EnvObservationType, RLObservationType
from srl.base.env.base import SpaceBase


class Processor(ABC):
    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
        original_env: object,
    ) -> Tuple[SpaceBase, EnvObservationType]:
        return env_observation_space, env_observation_type

    def process_observation(
        self,
        observation: np.ndarray,
        original_env: object,
    ) -> np.ndarray:
        return observation

    def process_rewards(
        self,
        rewards: List[float],
        original_env: object,
    ) -> List[float]:
        return rewards
