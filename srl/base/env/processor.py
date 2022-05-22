from abc import ABC
from typing import List, Tuple

from srl.base.define import (EnvObservation, EnvObservationType,
                             RLObservationType)
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
        observation: EnvObservation,
        original_env: object,
    ) -> EnvObservation:
        return observation

    def process_rewards(
        self,
        rewards: List[float],
        original_env: object,
    ) -> List[float]:
        return rewards
