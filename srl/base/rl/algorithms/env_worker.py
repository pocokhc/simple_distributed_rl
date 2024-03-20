from abc import abstractmethod
from typing import Tuple

from srl.base.define import EnvActionType, InfoType, RLActionType
from srl.base.env.env_run import EnvRun
from srl.base.rl.config import DummyRLConfig
from srl.base.rl.worker import RLWorker
from srl.base.rl.worker_run import WorkerRun


class EnvWorker(RLWorker):
    def __init__(self, **kwargs) -> None:
        super().__init__(DummyRLConfig(enable_action_decode=False))

    def call_on_reset(self, env: EnvRun) -> InfoType:
        return {}

    @abstractmethod
    def call_policy(self, env: EnvRun) -> Tuple[EnvActionType, InfoType]:
        raise NotImplementedError()

    def call_on_step(self, env: EnvRun) -> InfoType:
        return {}

    # -------------------------------

    def on_reset(self, worker: WorkerRun) -> InfoType:
        return self.call_on_reset(worker.env)

    def policy(self, worker: WorkerRun) -> Tuple[RLActionType, InfoType]:
        return self.call_policy(worker.env)  # type: ignore , EnvActionType ok

    def on_step(self, worker: WorkerRun) -> InfoType:
        return self.call_on_step(worker.env)
