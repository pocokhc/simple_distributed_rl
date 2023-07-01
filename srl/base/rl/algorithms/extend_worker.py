from typing import Optional

from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.base.rl.config import RLConfig
from srl.base.rl.worker_rl import RLWorker
from srl.base.rl.worker_run import WorkerRun


class ExtendWorker(RLWorker):
    def __init__(
        self,
        rl_worker: WorkerRun,
        config: Optional[RLConfig] = None,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ):
        super().__init__(config, parameter, remote_memory)
        self.rl_worker = rl_worker
        self.worker = self.rl_worker.worker
