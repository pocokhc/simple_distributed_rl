import logging
import pickle
from abc import ABC
from typing import TYPE_CHECKING, Union

from srl.base.define import DoneTypes, EnvObservationType
from srl.base.env.env_run import SpaceBase

if TYPE_CHECKING:
    from srl.base.env.env_run import EnvRun
    from srl.base.rl.config import RLConfig
logger = logging.getLogger(__name__)


ProcessorType = Union["ObservationProcessor", "EpisodeProcessor"]


class ObservationProcessor(ABC):
    def setup(self, env: "EnvRun", rl_config: "RLConfig"):
        pass

    def preprocess_observation_space(
        self,
        env_observation_space: SpaceBase,
        env: "EnvRun",
        rl_config: "RLConfig",
    ) -> SpaceBase:
        return env_observation_space

    def preprocess_observation(self, state: EnvObservationType, env: "EnvRun") -> EnvObservationType:
        return state

    def copy(self) -> "ObservationProcessor":
        o = self.__class__()

        for k, v in self.__dict__.items():
            try:
                setattr(o, k, pickle.loads(pickle.dumps(v)))
            except TypeError as e:
                logger.warning(f"'{k}' copy fail.({e})")

        return o


class EpisodeProcessor(ABC):
    def setup(self, env: "EnvRun", rl_config: "RLConfig"):
        pass

    def on_reset(self, env: "EnvRun"):
        pass

    def preprocess_reward(self, reward: float, env: "EnvRun") -> float:
        return reward

    def preprocess_done(self, done: DoneTypes, env: "EnvRun") -> DoneTypes:
        return done

    def copy(self) -> "EpisodeProcessor":
        o = self.__class__()

        for k, v in self.__dict__.items():
            try:
                setattr(o, k, pickle.loads(pickle.dumps(v)))
            except TypeError as e:
                logger.warning(f"'{k}' copy fail.({e})")

        return o
