import logging
from typing import Tuple

from srl.base.define import EnvActionType
from srl.base.env.base import EnvRun
from srl.base.env.spaces.continuous import ContinuousSpace
from srl.base.env.spaces.discrete import DiscreteSpace
from srl.base.rl.registration import register_worker
from srl.base.rl.worker import RuleBaseWorker, WorkerRun

logger = logging.getLogger(__name__)


register_worker("human", __name__ + ":Worker")


class Worker(RuleBaseWorker):
    def call_policy(self, env: EnvRun, worker: WorkerRun) -> Tuple[EnvActionType, dict]:
        if isinstance(env.action_space, DiscreteSpace):
            valid_actions = env.get_valid_actions()

            print("- select action -")
            for action in valid_actions:
                a1 = str(action)
                a2 = env.action_to_str(action)
                if a1 == a2:
                    print(f"{a1:>3s}")
                else:
                    print(f"{a1:>3s}: {a2}")
            for i in range(10):
                try:
                    action = int(input("> "))
                    if action in valid_actions:
                        break
                except Exception:
                    pass
                print(f"invalid action({10-i} times left)")
            else:
                raise ValueError()

        elif isinstance(env.action_space, ContinuousSpace):
            print(f"{env.action_space.low} - {env.action_space.high} >", end="")
            action = float(input())

        else:
            assert False, "Cannot input from terminal."

        return env.action_space.convert(action), {}
