from dataclasses import dataclass

from srl.base.define import RLTypes
from srl.base.rl.base import RLConfig
from srl.base.rl.remote_memory import ExperienceReplayBuffer


@dataclass
class MyConfig(RLConfig):
    # memoryの最大サイズ
    capacity: int = 1000

    def getName(self) -> str:
        return "MyConfig"

    @property
    def base_action_type(self) -> RLTypes:
        return RLTypes.DISCRETE

    @property
    def base_observation_type(self) -> RLTypes:
        return RLTypes.DISCRETE


class MyRemoteMemory(ExperienceReplayBuffer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: MyConfig = self.config

        # init を呼び出してメモリ容量をセットする。
        super().init(self.config.capacity)


# 実行例
memory = MyRemoteMemory(MyConfig())
memory.add(1)
memory.add(2)
memory.add(3)
memory.add(4)
dat = memory.sample(2)
print(dat)  # [3, 2]
