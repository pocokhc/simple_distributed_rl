import logging

from srl.base.define import EnvAction
from srl.base.env.base import EnvRun
from srl.base.rl.base import RuleBaseWorker, WorkerRun
from srl.base.rl.registration import register_worker

logger = logging.getLogger(__name__)


register_worker("random", __name__ + ":Worker")


class Worker(RuleBaseWorker):
    def call_on_reset(self, env: EnvRun, worker_run: WorkerRun) -> None:
        pass  # do nothing

    def call_policy(self, env: EnvRun, worker_run: WorkerRun) -> EnvAction:
        return env.sample(self.player_index)

    def call_render(self, env: EnvRun, worker_run: WorkerRun) -> None:
        pass  # do nothing
