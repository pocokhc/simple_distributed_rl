from abc import abstractmethod
from typing import Tuple

from srl.base.define import InfoType, RLActionType
from srl.base.rl.worker import WorkerBase
from srl.base.rl.worker_run import WorkerRun


class RLWorker(WorkerBase):
    def call_on_reset(self, worker: WorkerRun) -> InfoType:
        return {}

    @abstractmethod
    def call_policy(self, worker: WorkerRun) -> Tuple[RLActionType, InfoType]:
        raise NotImplementedError()

    def call_on_step(self, worker: WorkerRun) -> InfoType:
        return {}

    # -------------------------------

    def on_reset(self, worker: WorkerRun) -> InfoType:
        return self.call_on_reset(worker)

    def policy(self, worker: WorkerRun) -> Tuple[RLActionType, InfoType]:
        return self.call_policy(worker)

    def on_step(self, worker: WorkerRun) -> InfoType:
        return self.call_on_step(worker)
