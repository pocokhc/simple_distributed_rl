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

    def on_setup(self, worker, context) -> None:
        self.base_worker.on_setup(worker, context)

    def on_teardown(self, worker) -> None:
        self.base_worker.on_teardown(worker)

    def on_reset(self, worker) -> None:
        self.base_worker.on_reset(worker)

    def on_step(self, worker) -> None:
        self.base_worker.on_step(worker)

    def render_terminal(self, worker, **kwargs) -> None:
        self.base_worker.render_terminal(worker, **kwargs)

    def render_rgb_array(self, worker, **kwargs):
        return self.base_worker.render_rgb_array(worker, **kwargs)

    def _set_worker_run(self, worker: WorkerRun):
        super()._set_worker_run(worker)
        self.base_worker._set_worker_run(worker)
