from dataclasses import dataclass

import numpy as np

from srl.base.define import RLBaseActTypes, RLBaseObsTypes
from srl.base.rl.config import RLConfig
from srl.rl.memories.priority_experience_replay import (
    PriorityExperienceReplay,
    RLConfigComponentPriorityExperienceReplay,
)


@dataclass
class MyConfig(RLConfig, RLConfigComponentPriorityExperienceReplay):
    # RLConfig に加え、RLConfigComponentPriorityExperienceReplay も継承する
    # 順番は RLConfig -> RLConfigComponentPriorityExperienceReplay

    def get_name(self) -> str:
        return "MyConfig"

    def get_base_action_type(self) -> RLBaseActTypes:
        return RLBaseActTypes.DISCRETE

    def get_base_observation_type(self) -> RLBaseObsTypes:
        return RLBaseObsTypes.DISCRETE

    def get_framework(self) -> str:
        return ""


class MyRemoteMemory(PriorityExperienceReplay):
    pass


# --- select memory
config = MyConfig()
# config.set_replay_memory()
config.set_proportional_memory()
# config.set_rankbase_memory()

# --- run memory
memory = MyRemoteMemory(config)
memory.add(1, priority=1)
memory.add(2, priority=2)
memory.add(3, priority=3)
memory.add(4, priority=4)
batchs, weights, update_args = memory.sample(batch_size=1, step=0)
print(batchs)  # [2]
memory.update(update_args, np.array([5, 10, 15, 20, 11]))
