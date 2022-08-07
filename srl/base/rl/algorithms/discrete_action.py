import logging
from abc import abstractmethod
from typing import List

import numpy as np
from srl.base.define import DiscreteAction, EnvObservationType, Info, RLAction, RLActionType, RLObservation
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

    @property
    def action_num(self) -> int:
        return self._action_num


class DiscreteActionWorker(RLWorker):
    @abstractmethod
    def call_on_reset(
        self,
        state: np.ndarray,
        invalid_actions: List[RLAction],
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def call_policy(
        self,
        state: np.ndarray,
        invalid_actions: List[RLAction],
    ) -> DiscreteAction:
        raise NotImplementedError()

    @abstractmethod
    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[RLAction],
    ) -> Info:
        raise NotImplementedError()

    # --------------------------------------

    def _call_on_reset(self, state: RLObservation, env: EnvRun, worker: WorkerRun) -> None:
        self.call_on_reset(state, self.get_invalid_actions(env, worker))

    def _call_policy(self, state: RLObservation, env: EnvRun, worker: WorkerRun) -> RLAction:
        return self.call_policy(state, self.get_invalid_actions(env, worker))

    def _call_on_step(
        self,
        next_state: RLObservation,
        reward: float,
        done: bool,
        env: EnvRun,
        worker: WorkerRun,
    ) -> Info:
        return self.call_on_step(next_state, reward, done, self.get_invalid_actions(env, worker))
