import logging
from typing import Tuple

from srl.base.define import EnvActionType
from srl.base.env.env_run import EnvRun
from srl.base.rl.registration import register_worker
from srl.base.rl.worker import RuleBaseWorker, WorkerRun

logger = logging.getLogger(__name__)


register_worker("random", __name__ + ":Worker")


class Worker(RuleBaseWorker):
    def call_policy(self, env: EnvRun, worker: WorkerRun) -> Tuple[EnvActionType, dict]:
        return env.sample(), {}
