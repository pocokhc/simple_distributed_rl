import logging
from typing import Tuple

from srl.base.define import EnvAction
from srl.base.env.base import EnvRun
from srl.base.rl.registration import register_worker
from srl.base.rl.worker import RuleBaseWorker, WorkerRun

logger = logging.getLogger(__name__)


register_worker("random", __name__ + ":Worker")


class Worker(RuleBaseWorker):
    def call_on_reset(self, env: EnvRun, worker: WorkerRun) -> dict:
        return {}

    def call_policy(self, env: EnvRun, worker: WorkerRun) -> Tuple[EnvAction, dict]:
        return env.sample(), {}
