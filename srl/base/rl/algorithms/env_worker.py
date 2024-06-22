from abc import abstractmethod

from srl.base.define import EnvActionType, RLActionType
from srl.base.env.env_run import EnvRun
from srl.base.rl.config import DummyRLConfig
from srl.base.rl.worker import RLWorker
from srl.base.rl.worker_run import WorkerRun


class EnvWorker(RLWorker):
    def __init__(self, **kwargs) -> None:
        super().__init__(DummyRLConfig(enable_action_decode=False))

    def call_on_reset(self, env: EnvRun):
        pass

    @abstractmethod
    def call_policy(self, env: EnvRun) -> EnvActionType:
        raise NotImplementedError()

    def call_on_step(self, env: EnvRun):
        pass

    # -------------------------------

    def on_reset(self, worker: WorkerRun):
        self.call_on_reset(worker.env)

    def policy(self, worker: WorkerRun) -> RLActionType:
        return self.call_policy(worker.env)  # type: ignore , EnvActionType ok

    def on_step(self, worker: WorkerRun):
        self.call_on_step(worker.env)
