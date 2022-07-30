from abc import abstractmethod

from srl.base.define import Info, RLAction, RLObservation
from srl.base.env.base import EnvRun
from srl.base.rl.base import RLWorker, WorkerRun


class ModelBaseWorker(RLWorker):
    @abstractmethod
    def call_on_reset(self, state: RLObservation, env: EnvRun, worker: WorkerRun) -> None:
        raise NotImplementedError()

    @abstractmethod
    def call_policy(self, state: RLObservation, env: EnvRun, worker: WorkerRun) -> RLAction:
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

    def _call_on_reset(self, state: RLObservation, env: EnvRun, worker: WorkerRun) -> None:
        self.call_on_reset(state, env, worker)

    def _call_policy(self, state: RLObservation, env: EnvRun, worker: WorkerRun) -> RLAction:
        return self.call_policy(state, env, worker)

    def _call_on_step(
        self,
        next_state: RLObservation,
        reward: float,
        done: bool,
        env: EnvRun,
        worker: WorkerRun,
    ) -> Info:
        return self.call_on_step(next_state, reward, done, env, worker)
