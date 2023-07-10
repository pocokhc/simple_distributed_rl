from typing import Optional

from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.base.rl.config import RLConfig
from srl.base.rl.worker import WorkerBase
from srl.base.rl.worker_rl import RLWorker
from srl.base.rl.worker_run import WorkerRun


class ExtendWorker(RLWorker):
    def __init__(
        self,
        worker: WorkerBase,
        config: Optional[RLConfig] = None,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ):
        super().__init__(config, parameter, remote_memory)
        self.base_worker = worker

    def _set_worker_run(self, worker: WorkerRun):
        super()._set_worker_run(worker)
        self.base_worker._set_worker_run(worker)
