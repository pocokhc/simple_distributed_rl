from abc import ABC
from typing import TYPE_CHECKING, Tuple

from srl.base.define import EnvObservationType, EnvObservationTypes
from srl.base.env.env_run import SpaceBase

if TYPE_CHECKING:
    from srl.base.env.env_run import EnvRun
    from srl.base.rl.config import RLConfig


class Processor(ABC):
    """
    Preprocess information about the environment. (for RL)
    """

    def preprocess_observation_space(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationTypes,
        env: "EnvRun",
        rl_config: "RLConfig",
    ) -> Tuple[SpaceBase, EnvObservationTypes]:
        return env_observation_space, env_observation_type

    def preprocess_observation(
        self,
        observation: EnvObservationType,
        env: "EnvRun",
    ) -> EnvObservationType:
        return observation

    def preprocess_reward(
        self,
        reward: float,
        env: "EnvRun",
    ) -> float:
        return reward
