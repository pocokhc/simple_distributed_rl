import logging
from typing import Tuple

from srl.base.define import EnvActionType, RLTypes
from srl.base.rl.base import RLWorker
from srl.base.rl.registration import register_rulebase
from srl.base.rl.worker_run import WorkerRun
from srl.base.spaces import ContinuousSpace

logger = logging.getLogger(__name__)


register_rulebase("human", __name__ + ":Worker")


class Worker(RLWorker):
    def policy(self, worker: WorkerRun) -> Tuple[EnvActionType, dict]:
        if self.config.action_type == RLTypes.DISCRETE:
            invalid_actions = worker.get_invalid_actions()
            action_num = self.config.action_space.n

            print("- select action -")
            arr = []
            for action in range(self.config.action_space.n):
                if action in invalid_actions:
                    continue
                a1 = str(action)
                a2 = worker.env.action_to_str(action)
                if a1 == a2:
                    arr.append(f"{a1}")
                else:
                    arr.append(f"{a1}({a2})")
            print(" ".join(arr))
            for i in range(10):
                try:
                    action = int(input("> "))
                    if (action not in invalid_actions) and (0 <= action < action_num):
                        break
                except Exception:
                    pass
                print(f"invalid action({10-i} times left)")
            else:
                raise ValueError()

        elif isinstance(self.config.action_type, ContinuousSpace):
            print(f"{self.config.action_space.low} - {self.config.action_space.high} >", end="")
            action = float(input())

        else:
            assert False, "Cannot input from terminal."

        return self.config.action_space.convert(action), {}
