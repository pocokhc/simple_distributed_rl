import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from srl.base.define import (
    ContinuousActionType,
    EnvObservationTypes,
    InfoType,
    RLActionType,
    RLActionTypes,
    RLObservationType,
)
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.rl.config import RLConfig
from srl.base.rl.worker import RLWorker, WorkerRun

logger = logging.getLogger(__name__)


@dataclass
class ContinuousActionConfig(RLConfig):
    @property
    def action_type(self) -> RLActionTypes:
        return RLActionTypes.CONTINUOUS

    def set_config_by_env(
        self,
        env: EnvRun,
        env_action_space: SpaceBase,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationTypes,
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
    def call_on_reset(self, state: np.ndarray) -> InfoType:
        raise NotImplementedError()

    @abstractmethod
    def call_policy(self, state: np.ndarray) -> Tuple[ContinuousActionType, InfoType]:
        raise NotImplementedError()

    @abstractmethod
    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> InfoType:
        raise NotImplementedError()

    # ----------------------------------

    def _call_on_reset(self, state: RLObservationType, env: EnvRun, worker: WorkerRun) -> InfoType:
        return self.call_on_reset(state)

    def _call_policy(self, state: RLObservationType, env: EnvRun, worker: WorkerRun) -> Tuple[RLActionType, InfoType]:
        return self.call_policy(state)

    def _call_on_step(
        self,
        next_state: RLObservationType,
        reward: float,
        done: bool,
        env: EnvRun,
        worker: WorkerRun,
    ) -> InfoType:
        return self.call_on_step(next_state, reward, done)
