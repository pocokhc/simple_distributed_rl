import logging
from abc import abstractmethod

import numpy as np
from srl.base.define import (
    ContinuousAction,
    EnvObservationType,
    Info,
    RLAction,
    RLActionType,
    RLObservation,
    RLObservationType,
)
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.rl.base import RLConfig, RLWorker, WorkerRun

logger = logging.getLogger(__name__)


class ContinuousActionConfig(RLConfig):
    @property
    def action_type(self) -> RLActionType:
        return RLActionType.CONTINUOUS

    def _set_config_by_env(
        self,
        env: EnvRun,
        env_action_space: SpaceBase,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
    ) -> None:
        n, low, high = env_action_space.get_action_continuous_info()
        self._action_num = n
        self._action_low = low
        self._action_high = high

    @property
    def action_num(self) -> int:
        return self._action_num

    @property
    def action_low(self) -> np.ndarray:
        return self._action_low

    @property
    def action_high(self) -> np.ndarray:
        return self._action_high


class ContinuousActionWorker(RLWorker):
    @abstractmethod
    def call_on_reset(self, state: np.ndarray) -> None:
        raise NotImplementedError()

    @abstractmethod
    def call_policy(self, state: np.ndarray) -> ContinuousAction:
        raise NotImplementedError()

    @abstractmethod
    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> Info:
        raise NotImplementedError()

    # ----------------------------------

    def _call_on_reset(self, state: RLObservation, env: EnvRun, worker: WorkerRun) -> None:
        self.call_on_reset(state)

    def _call_policy(self, state: RLObservation, env: EnvRun, worker: WorkerRun) -> RLAction:
        return self.call_policy(state)

    def _call_on_step(
        self,
        next_state: RLObservation,
        reward: float,
        done: bool,
        env: EnvRun,
        worker: WorkerRun,
    ) -> Info:
        return self.call_on_step(next_state, reward, done)
