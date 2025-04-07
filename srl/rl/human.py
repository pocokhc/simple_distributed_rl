import logging
from dataclasses import dataclass

from srl.base.define import RLBaseActTypes, RLBaseObsTypes
from srl.base.rl.config import RLConfig
from srl.base.rl.registration import register_rulebase
from srl.base.rl.worker import RLWorker
from srl.base.spaces.discrete import DiscreteSpace

logger = logging.getLogger(__name__)


@dataclass
class Config(RLConfig):
    def get_base_action_type(self) -> RLBaseActTypes:
        return RLBaseActTypes.DISCRETE

    def get_base_observation_type(self) -> RLBaseObsTypes:
        return RLBaseObsTypes.NONE

    def get_name(self) -> str:
        return "human"


register_rulebase(Config(), __name__ + ":Worker")


class Worker(RLWorker):
    def policy(self, worker):
        assert isinstance(self.config.action_space, DiscreteSpace)

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
            print(f"invalid action({10 - i} times left)")
        else:
            raise ValueError()

        return self.config.action_space.sanitize(action)
