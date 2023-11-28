from abc import ABC
from typing import TYPE_CHECKING, Tuple

from srl.base.define import EnvObservationType, EnvObservationTypes
from srl.base.env.env_run import SpaceBase

if TYPE_CHECKING:
    from srl.base.env.env_run import EnvRun
    from srl.base.rl.config import RLConfig


class Processor(ABC):
    def on_reset(self, env: "EnvRun"):
        pass

    def setup(self, env: "EnvRun", rl_config: "RLConfig"):
        pass

    def preprocess_observation_space(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationTypes,
        env: "EnvRun",
        rl_config: "RLConfig",
    ) -> Tuple[SpaceBase, EnvObservationTypes]:
        return env_observation_space, env_observation_type

    def preprocess_observation(self, observation: EnvObservationType, env: "EnvRun") -> EnvObservationType:
        return observation

    def preprocess_reward(self, reward: float, env: "EnvRun") -> float:
        return reward

    def preprocess_done(self, done: bool, env: "EnvRun") -> bool:
        return done
