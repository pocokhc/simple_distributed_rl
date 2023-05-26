import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from srl.base.define import EnvObservationTypes, InfoType, RLActionType, RLObservationType, RLTypes
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.rl.base import RLConfig
from srl.base.rl.worker import RLWorker, WorkerRun

logger = logging.getLogger(__name__)


@dataclass
class AnyActionConfig(RLConfig):
    @property
    def action_type(self) -> RLTypes:
        return RLTypes.ANY

    def set_config_by_env(
        self,
        env: EnvRun,
        env_action_space: SpaceBase,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationTypes,
    ) -> None:
        if env_action_space.rl_type == RLTypes.DISCRETE:
            self._action_num = env_action_space.n
            self._action_low = np.ndarray(0)
            self._action_high = np.ndarray(self._action_num - 1)
            self._env_action_type = RLTypes.DISCRETE
        else:
            # ANYの場合もCONTINUOUS
            self._action_num = env_action_space.list_size
            self._action_low = np.array(env_action_space.list_low)
            self._action_high = np.array(env_action_space.list_high)
            self._env_action_type = RLTypes.CONTINUOUS

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
    def env_action_type(self) -> RLTypes:
        return self._env_action_type


class AnyActionWorker(RLWorker):
    @abstractmethod
    def call_on_reset(
        self,
        state: np.ndarray,
        invalid_actions: List[int],
    ) -> InfoType:
        raise NotImplementedError()

    @abstractmethod
    def call_policy(
        self,
        state: np.ndarray,
        invalid_actions: List[int],
    ) -> Tuple[RLActionType, InfoType]:
        raise NotImplementedError()

    @abstractmethod
    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ) -> InfoType:
        raise NotImplementedError()

    # --------------------------------------

    def _call_on_reset(self, state: RLObservationType, env: EnvRun, worker: WorkerRun) -> InfoType:
        return self.call_on_reset(state, self.get_invalid_actions())

    def _call_policy(self, state: RLObservationType, env: EnvRun, worker: WorkerRun) -> Tuple[RLActionType, InfoType]:
        return self.call_policy(state, self.get_invalid_actions())

    def _call_on_step(
        self,
        next_state: RLObservationType,
        reward: float,
        done: bool,
        env: EnvRun,
        worker: WorkerRun,
    ) -> InfoType:
        return self.call_on_step(next_state, reward, done, self.get_invalid_actions())
