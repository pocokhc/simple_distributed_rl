from abc import ABC
from typing import Tuple

from srl.base.define import EnvObservationType, EnvObservationTypes, RLTypes
from srl.base.env.env_run import EnvRun, SpaceBase


class Processor(ABC):
    """
    Preprocess information about the environment. (for RL)
    """

    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationTypes,
        rl_observation_type: RLTypes,
        env: EnvRun,
    ) -> Tuple[SpaceBase, EnvObservationTypes]:
        return env_observation_space, env_observation_type

    def process_observation(
        self,
        observation: EnvObservationType,
        env: EnvRun,
    ) -> EnvObservationType:
        return observation

    def process_reward(
        self,
        reward: float,
        env: EnvRun,
    ) -> float:
        return reward
