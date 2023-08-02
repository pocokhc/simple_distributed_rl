from dataclasses import dataclass

from srl.base.define import RLTypes
from srl.base.rl.base import RLConfig
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, ExperienceReplayBufferConfig


@dataclass
class MyConfig(RLConfig, ExperienceReplayBufferConfig):
    # RLConfig に加え、ExperienceReplayBufferConfig も継承する
    # (順番は RLConfig -> ExperienceReplayBufferConfig )

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


class MyRemoteMemory(ExperienceReplayBuffer):
    pass


# 実行例
memory = MyRemoteMemory(MyConfig())
memory.add(1)
memory.add(2)
memory.add(3)
memory.add(4)
dat = memory.sample(2)
print(dat)  # [3, 2]
