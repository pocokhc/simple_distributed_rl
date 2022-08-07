from abc import ABC
from typing import Tuple

from srl.base.define import EnvObservation, EnvObservationType, RLObservationType
from srl.base.env.base import EnvRun, SpaceBase


class Processor(ABC):
    """
    Preprocess information about the environment. (for RL)
    """

    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
        env: EnvRun,
    ) -> Tuple[SpaceBase, EnvObservationType]:
        return env_observation_space, env_observation_type

    def process_observation(
        self,
        observation: EnvObservation,
        env: EnvRun,
    ) -> EnvObservation:
        return observation

    def process_reward(
        self,
        reward: float,
        env: EnvRun,
    ) -> float:
        return reward
