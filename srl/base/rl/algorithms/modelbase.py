from abc import abstractmethod
from typing import Tuple

import numpy as np

from srl.base.define import InfoType, RLActionType, RLObservationType
from srl.base.env.env_run import EnvRun
from srl.base.rl.worker import RLWorker
from srl.base.rl.worker_run import WorkerRun


class ModelBaseWorker(RLWorker):
    @abstractmethod
    def call_on_reset(self, state: np.ndarray, env: EnvRun, worker: WorkerRun) -> InfoType:
        raise NotImplementedError()

    @abstractmethod
    def call_policy(self, state: np.ndarray, env: EnvRun, worker: WorkerRun) -> Tuple[RLActionType, InfoType]:
        raise NotImplementedError()

    @abstractmethod
    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        env: EnvRun,
        worker: WorkerRun,
    ) -> InfoType:
        raise NotImplementedError()

    # --------------------------

    def _call_on_reset(self, state: RLObservationType, env: EnvRun, worker: WorkerRun) -> InfoType:
        return self.call_on_reset(state, env, worker)

    def _call_policy(self, state: RLObservationType, env: EnvRun, worker: WorkerRun) -> Tuple[RLActionType, InfoType]:
        return self.call_policy(state, env, worker)

    def _call_on_step(
        self,
        next_state: RLObservationType,
        reward: float,
        done: bool,
        env: EnvRun,
        worker: WorkerRun,
    ) -> InfoType:
        return self.call_on_step(next_state, reward, done, env, worker)
