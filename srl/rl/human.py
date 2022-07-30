import logging

from srl.base.define import EnvAction
from srl.base.env.base import EnvRun
from srl.base.rl.base import RuleBaseWorker, WorkerRun
from srl.base.rl.registration import register_worker

logger = logging.getLogger(__name__)


register_worker("human", __name__ + ":Worker")


class Worker(RuleBaseWorker):
    def call_on_reset(self, env: EnvRun, worker_run: WorkerRun) -> None:
        pass  # do nothing

    def call_policy(self, env: EnvRun, worker_run: WorkerRun) -> EnvAction:
        invalid_actions = env.get_invalid_actions(self.player_index)
        action_num = env.action_space.get_action_discrete_info()

        actions = [a for a in range(action_num) if a not in invalid_actions]
        print(f"select action: {actions}")
        for i in range(10):
            try:
                action = int(input("> "))
                if action in actions:
                    break
            except Exception:
                print(f"invalid action({10-i} times left)")

        action = env.action_space.action_discrete_decode(action)
        return action

    def render_terminal(self, env, worker, **kwargs) -> None:
        pass  # do nothing
