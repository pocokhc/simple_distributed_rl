import logging
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from srl.base.define import EnvObservationType, RLActionType, RLObservationType
from srl.base.rl.base import RLConfig, RLWorker

logger = logging.getLogger(__name__)


class DiscreteActionConfig(RLConfig):
    @property
    def action_type(self) -> RLActionType:
        return RLActionType.DISCRETE

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.CONTINUOUS

    def _set_config_by_env(self, env: "srl.base.rl.env_for_rl.EnvForRL") -> None:
        self._nb_actions = env.action_space.n
        self._env_observation_shape = env.observation_space.shape
        self._env_observation_type = env.observation_type

    @property
    def nb_actions(self) -> int:
        return self._nb_actions

    @property
    def env_observation_shape(self) -> tuple:
        return self._env_observation_shape

    @property
    def env_observation_type(self) -> EnvObservationType:
        return self._env_observation_type


class DiscreteActionWorker(RLWorker):
    @abstractmethod
    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> None:
        raise NotImplementedError()

    def on_reset(
        self,
        state: np.ndarray,
        invalid_actions: List[int],
        env: "srl.base.rl.env_for_rl.EnvForRL",
    ) -> None:
        self.call_on_reset(state, invalid_actions)

    @abstractmethod
    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> int:
        raise NotImplementedError()

    def policy(
        self,
        state: np.ndarray,
        invalid_actions: List[int],
        env: "srl.base.rl.env_for_rl.EnvForRL",
    ) -> Any:
        return self.call_policy(state, invalid_actions)

    @abstractmethod
    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ) -> Dict[str, Union[float, int]]:  # info
        raise NotImplementedError()

    def on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
        env: "srl.base.rl.env_for_rl.EnvForRL",
    ) -> Dict[str, Union[float, int]]:  # info
        return self.call_on_step(next_state, reward, done, next_invalid_actions)


if __name__ == "__main__":
    pass
