from dataclasses import dataclass

import numpy as np

from srl.base.define import RLTypes
from srl.base.rl.base import RLConfig
from srl.rl.memories.priority_experience_replay import PriorityExperienceReplay, PriorityExperienceReplayConfig


@dataclass
class MyConfig(RLConfig, PriorityExperienceReplayConfig):
    # RLConfig PriorityExperienceReplayConfig も継承する
    # (順番は RLConfig -> PriorityExperienceReplayConfig )

    def getName(self) -> str:
        return "MyConfig"

    @property
    def base_action_type(self) -> RLTypes:
        return RLTypes.DISCRETE

    @property
    def base_observation_type(self) -> RLTypes:
        return RLTypes.DISCRETE

    def get_use_framework(self) -> str:
        return ""


class MyRemoteMemory(PriorityExperienceReplay):
    pass


# 実行例
memory = MyRemoteMemory(MyConfig())
memory.add(1, 1)
memory.add(2, 2)
memory.add(3, 3)
memory.add(4, 4)
indices, batchs, weights = memory.sample(batch_size=1, step=5)
print(batchs)  # [2]
memory.update(indices, batchs, np.array([5, 10, 15, 20, 11]))
