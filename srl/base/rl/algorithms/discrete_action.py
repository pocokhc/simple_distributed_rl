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
    RLObservation,
    RLObservationType,
)
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.rl.base import RLConfig, RLWorker, WorkerRun

logger = logging.getLogger(__name__)


class DiscreteActionConfig(RLConfig):
    @property
    def action_type(self) -> RLActionType:
        return RLActionType.DISCRETE

    def _set_config_by_env(
        self,
        env: EnvRun,
        env_action_space: SpaceBase,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
    ) -> None:
        self._action_num = env_action_space.get_action_discrete_info()

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
    def action_num(self) -> int:
        return self._action_num

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
    def call_on_reset(
        self,
        state: np.ndarray,
        invalid_actions: List[RLInvalidAction],
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def call_policy(
        self,
        state: np.ndarray,
        invalid_actions: List[RLInvalidAction],
    ) -> DiscreteAction:
        raise NotImplementedError()

    @abstractmethod
    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[RLInvalidAction],
    ) -> Info:
        raise NotImplementedError()

    @abstractmethod
    def call_render(self, env: EnvRun) -> Info:
        raise NotImplementedError()

    # --------------------------------------

    def _call_on_reset(self, state: RLObservation, env: EnvRun, worker: WorkerRun) -> None:
        self.call_on_reset(state, self.get_invalid_actions(env))

    def _call_policy(self, state: RLObservation, env: EnvRun, worker: WorkerRun) -> RLAction:
        return self.call_policy(state, self.get_invalid_actions(env))

    def _call_on_step(
        self,
        next_state: RLObservation,
        reward: float,
        done: bool,
        env: EnvRun,
        worker: WorkerRun,
    ) -> Info:
        return self.call_on_step(next_state, reward, done, self.get_invalid_actions(env))

    def _call_render(self, env: EnvRun, worker: WorkerRun) -> Info:
        return self.call_render(env)
