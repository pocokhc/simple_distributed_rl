from abc import abstractmethod
from typing import Optional

from srl.base.define import RLActionType
from srl.base.rl.config import RLConfig
from srl.base.rl.memory import IRLMemoryWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.worker import RLWorker
from srl.base.rl.worker_run import WorkerRun


class ExtendWorker(RLWorker):
    def __init__(
        self,
        worker: RLWorker,
        config: RLConfig,
        parameter: Optional[RLParameter] = None,
        memory: Optional[IRLMemoryWorker] = None,
    ):
        super().__init__(config, parameter, memory)
        self.base_worker = worker

    def _set_worker_run(self, worker: WorkerRun):
        super()._set_worker_run(worker)
        self.base_worker._set_worker_run(worker)

    def call_on_reset(self, worker: WorkerRun):
        pass

    @abstractmethod
    def call_policy(self, worker: WorkerRun) -> RLActionType:
        raise NotImplementedError()

    def call_on_step(self, worker: WorkerRun):
        pass

    # -------------------------------

    def on_reset(self, worker: WorkerRun):
        self.call_on_reset(worker)

    def policy(self, worker: WorkerRun) -> RLActionType:
        return self.call_policy(worker)

    def on_step(self, worker: WorkerRun):
        self.call_on_step(worker)
