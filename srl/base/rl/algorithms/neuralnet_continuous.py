import logging
from abc import abstractmethod

import numpy as np
from srl.base.define import Action, ContinuousAction, EnvObservationType, Info, RLActionType, RLObservationType
from srl.base.env.base import EnvBase, SpaceBase
from srl.base.rl.base import RLConfig, RLWorker

logger = logging.getLogger(__name__)


class ContinuousActionConfig(RLConfig):
    @property
    def action_type(self) -> RLActionType:
        return RLActionType.CONTINUOUS

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
        n, low, high = env_action_space.get_action_continuous_info()
        self._action_num = n
        self._action_low = low
        self._action_high = high
        shape, _, _ = env_observation_space.get_observation_continuous_info()
        self._env_observation_shape = shape

    @property
    def action_num(self) -> int:
        return self._action_num

    @property
    def action_low(self) -> np.ndarray:
        return self._action_low

    @property
    def action_high(self) -> np.ndarray:
        return self._action_high

    @property
    def env_observation_shape(self) -> tuple:
        return self._env_observation_shape


class ContinuousActionWorker(RLWorker):
    @abstractmethod
    def call_on_reset(self, state: np.ndarray) -> None:
        raise NotImplementedError()

    def _on_reset(
        self,
        state: np.ndarray,
        player_index: int,
        env: EnvBase,
    ) -> None:
        self.call_on_reset(state)

    @abstractmethod
    def call_policy(self, state: np.ndarray) -> ContinuousAction:
        raise NotImplementedError()

    def _policy(
        self,
        state: np.ndarray,
        player_index: int,
        env: EnvBase,
    ) -> Action:
        return self.call_policy(state)

    @abstractmethod
    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
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
        return self.call_on_step(next_state, reward, done)
