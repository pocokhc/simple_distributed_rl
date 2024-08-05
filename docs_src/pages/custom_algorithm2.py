from dataclasses import dataclass

from srl.base.define import RLBaseActTypes, RLBaseObsTypes
from srl.base.rl.config import RLConfig
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, RLConfigComponentExperienceReplayBuffer


@dataclass
class MyConfig(
    RLConfig,
    RLConfigComponentExperienceReplayBuffer,
):
    # RLConfig に加え、RLConfigComponentExperienceReplayBuffer も継承する
    # 順番は RLConfig -> RLConfigComponentExperienceReplayBuffer

    def get_name(self) -> str:
        return "MyConfig"

    def get_base_action_type(self) -> RLBaseActTypes:
        return RLBaseActTypes.DISCRETE

    def get_base_observation_type(self) -> RLBaseObsTypes:
        return RLBaseObsTypes.DISCRETE

    def get_framework(self) -> str:
        return ""


class MyRemoteMemory(ExperienceReplayBuffer):
    pass


# 実行例
memory = MyRemoteMemory(MyConfig())
memory.add(1)
memory.add(2)
memory.add(3)
memory.add(4)
dat = memory.sample(batch_size=2)
print(dat)  # [3, 2]
