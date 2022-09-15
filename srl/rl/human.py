import logging
from typing import Tuple

from srl.base.define import EnvAction
from srl.base.env.base import EnvRun
from srl.base.rl.registration import register_worker
from srl.base.rl.worker import RuleBaseWorker, WorkerRun

logger = logging.getLogger(__name__)


register_worker("human", __name__ + ":Worker")


class Worker(RuleBaseWorker):
    def call_on_reset(self, env: EnvRun, worker: WorkerRun) -> dict:
        return {}

    def call_policy(self, env: EnvRun, worker: WorkerRun) -> Tuple[EnvAction, dict]:
        valid_actions = env.get_valid_actions()

        print(f"select action: {valid_actions}")
        for i in range(10):
            try:
                action = int(input("> "))
                if action in valid_actions:
                    break
            except Exception:
                print(f"invalid action({10-i} times left)")
        else:
            raise ValueError()

        action = env.action_space.action_discrete_decode(action)
        return action, {}
