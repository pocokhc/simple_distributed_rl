import logging
from abc import abstractmethod
from typing import List

import numpy as np
from srl.base.define import (
    DiscreteAction,
    EnvObservationType,
    Info,
    RLAction,
    RLActionType,
    RLInvalidAction,
    RLObservationType,
)
from srl.base.env.base import EnvBase, SpaceBase
from srl.base.rl.base import RLConfig, RLWorker

logger = logging.getLogger(__name__)


class DiscreteActionConfig(RLConfig):
    @property
    def action_type(self) -> RLActionType:
        return RLActionType.DISCRETE

    def _set_config_by_env(
        self,
        env: EnvBase,
        env_action_space: SpaceBase,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
    ) -> None:
        self._nb_actions = env_action_space.get_action_discrete_info()

        if self.observation_type == RLObservationType.DISCRETE:
            shape, low, high = env_observation_space.get_observation_discrete_info()
        elif self.observation_type == RLObservationType.CONTINUOUS:
            shape, low, high = env_observation_space.get_observation_continuous_info()
        else:
            shape = (0,)
            low = np.array([0])
            high = np.array([0])
        self._observation_shape = shape
        self._observation_low = low
        self._observation_high = high

    @property
    def nb_actions(self) -> int:
        return self._nb_actions

    @property
    def observation_shape(self) -> tuple:
        return self._observation_shape

    @property
    def observation_low(self) -> np.ndarray:
        return self._observation_low

    @property
    def observation_high(self) -> np.ndarray:
        return self._observation_high


class DiscreteActionWorker(RLWorker):
    @abstractmethod
    def call_on_reset(self, state: np.ndarray, invalid_actions: List[RLInvalidAction]) -> None:
        raise NotImplementedError()

    def _on_reset(
        self,
        state: np.ndarray,
        player_index: int,
        env: EnvBase,
    ) -> None:
        self.call_on_reset(state, self.get_invalid_actions(env, player_index))

    @abstractmethod
    def call_policy(self, state: np.ndarray, invalid_actions: List[RLInvalidAction]) -> DiscreteAction:
        raise NotImplementedError()

    def _policy(
        self,
        state: np.ndarray,
        player_index: int,
        env: EnvBase,
    ) -> RLAction:
        return self.call_policy(state, self.get_invalid_actions(env, player_index))

    @abstractmethod
    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[RLInvalidAction],
    ) -> Info:
        raise NotImplementedError()

    def _on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        player_index: int,
        env: EnvBase,
    ) -> Info:
        return self.call_on_step(next_state, reward, done, self.get_invalid_actions(env, player_index))
