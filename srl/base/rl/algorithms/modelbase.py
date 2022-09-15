import warnings
from abc import abstractmethod
from typing import Tuple

from srl.base.define import Info, RLAction, RLObservation
from srl.base.env.base import EnvRun
from srl.base.rl.worker import RLWorker, WorkerRun


class ModelBaseWorker(RLWorker):
    @abstractmethod
    def call_on_reset(self, state: RLObservation, env: EnvRun, worker: WorkerRun) -> Info:
        raise NotImplementedError()

    @abstractmethod
    def call_policy(self, state: RLObservation, env: EnvRun, worker: WorkerRun) -> Tuple[RLAction, Info]:
        raise NotImplementedError()

    @abstractmethod
    def call_on_step(
        self,
        next_state: RLObservation,
        reward: float,
        done: bool,
        env: EnvRun,
        worker: WorkerRun,
    ) -> Info:
        raise NotImplementedError()

    # --------------------------

    def _call_on_reset(self, state: RLObservation, env: EnvRun, worker: WorkerRun) -> Info:
        _t = self.call_on_reset(state, env, worker)
        if _t is None:
            warnings.warn("The return value of call_on_reset has changed from None to info.", DeprecationWarning)
            return {}
        return _t

    def _call_policy(self, state: RLObservation, env: EnvRun, worker: WorkerRun) -> Tuple[RLAction, Info]:
        action = self.call_policy(state, env, worker)
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
        return self.call_on_step(next_state, reward, done, env, worker)
