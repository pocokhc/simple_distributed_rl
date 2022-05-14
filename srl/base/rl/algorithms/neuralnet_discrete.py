import logging
from abc import abstractmethod
from typing import List

import numpy as np
from srl.base.define import Action, DiscreteAction, EnvObservationType, Info, RLActionType, RLObservationType
from srl.base.env.base import EnvBase, SpaceBase
from srl.base.rl.base import RLConfig, RLWorker

logger = logging.getLogger(__name__)


class DiscreteActionConfig(RLConfig):
    @property
    def action_type(self) -> RLActionType:
        return RLActionType.DISCRETE

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.CONTINUOUS

    def _set_config_by_env(
        self,
        env: EnvBase,
        env_action_space: SpaceBase,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
    ) -> None:
        self._nb_actions = env_action_space.get_action_discrete_info()
        shape, _, _ = env_observation_space.get_observation_continuous_info()
        self._env_observation_shape = shape

    @property
    def nb_actions(self) -> int:
        return self._nb_actions

    @property
    def env_observation_shape(self) -> tuple:
        return self._env_observation_shape


class DiscreteActionWorker(RLWorker):
    @abstractmethod
    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> None:
        raise NotImplementedError()

    def _on_reset(
        self,
        state: np.ndarray,
        player_index: int,
        env: EnvBase,
    ) -> None:
        self.call_on_reset(state, env.get_invalid_actions(player_index))

    @abstractmethod
    def call_policy(self, state: np.ndarray, invalid_actions: List[DiscreteAction]) -> int:
        raise NotImplementedError()

    def _policy(
        self,
        state: np.ndarray,
        player_index: int,
        env: EnvBase,
    ) -> Action:
        return self.call_policy(state, env.get_invalid_actions(player_index))

    @abstractmethod
    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[DiscreteAction],
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
        return self.call_on_step(next_state, reward, done, env.get_invalid_actions(player_index))
