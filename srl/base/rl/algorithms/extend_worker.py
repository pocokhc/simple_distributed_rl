from typing import Optional

from srl.base.rl.config import RLConfig
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.worker import RLWorker
from srl.base.rl.worker_run import WorkerRun


class ExtendWorker(RLWorker):
    def __init__(
        self,
        worker: RLWorker,
        config: RLConfig,
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
    ):
        super().__init__(config, parameter, memory)
        self.base_worker = worker

    def _set_worker_run(self, worker: WorkerRun):
        super()._set_worker_run(worker)
        self.base_worker._set_worker_run(worker)
