from dataclasses import dataclass, field
from typing import cast

import numpy as np

from srl.base.define import RLTypes
from srl.base.rl.base import RLConfig
from srl.base.rl.memory import IPriorityMemoryConfig
from srl.base.rl.remote_memory.priority_experience_replay import PriorityExperienceReplay
from srl.rl.memories.config import ReplayMemoryConfig


@dataclass
class MyConfig(RLConfig):
    # PriorityExperienceReplay用のパラメータ
    # 型は IPriorityMemoryConfig を取る(IPriorityMemoryConfigは後述)
    memory: IPriorityMemoryConfig = field(default_factory=lambda: ReplayMemoryConfig())

    def getName(self) -> str:
        return "MyConfig"

    @property
    def action_type(self) -> RLTypes:
        return RLTypes.DISCRETE

    @property
    def observation_type(self) -> RLTypes:
        return RLTypes.DISCRETE


class MyRemoteMemory(PriorityExperienceReplay):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(MyConfig, self.config)

        # IPriorityMemoryConfig を init に渡す
        super().init(self.config.memory)


# 実行例
memory = MyRemoteMemory(MyConfig())
memory.add(1, 1)
memory.add(2, 2)
memory.add(3, 3)
memory.add(4, 4)
indices, batchs, weights = memory.sample(batch_size=1, step=5)
print(batchs)  # [2]
memory.update(indices, batchs, np.array([5, 10, 15, 20, 11]))
