import logging
import warnings
from abc import abstractmethod
from typing import Tuple

import numpy as np
from srl.base.define import ContinuousAction, EnvObservationType, Info, RLAction, RLActionType, RLObservation
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.rl.config import RLConfig
from srl.base.rl.worker import RLWorker, WorkerRun

logger = logging.getLogger(__name__)


class ContinuousActionConfig(RLConfig):
    @property
    def action_type(self) -> RLActionType:
        return RLActionType.CONTINUOUS

    def set_config_by_env(
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
    def call_on_reset(self, state: np.ndarray) -> Info:
        raise NotImplementedError()

    @abstractmethod
    def call_policy(self, state: np.ndarray) -> Tuple[ContinuousAction, Info]:
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

    def _call_on_reset(self, state: RLObservation, env: EnvRun, worker: WorkerRun) -> Info:
        _t = self.call_on_reset(state)
        if _t is None:
            warnings.warn("The return value of call_on_reset has changed from None to info.", DeprecationWarning)
            return {}
        return _t

    def _call_policy(self, state: RLObservation, env: EnvRun, worker: WorkerRun) -> Tuple[RLAction, Info]:
        action = self.call_policy(state)
        if isinstance(action, tuple) and len(action) == 2 and isinstance(action[1], dict):
            action, info = action
        else:
            warnings.warn(
                "The return value of call_policy has changed from action to (action, info).", DeprecationWarning
            )
            info = {}
        return action, info

    def _call_on_step(
        self,
        next_state: RLObservation,
        reward: float,
        done: bool,
        env: EnvRun,
        worker: WorkerRun,
    ) -> Info:
        return self.call_on_step(next_state, reward, done)
