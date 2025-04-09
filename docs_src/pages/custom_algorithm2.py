from dataclasses import dataclass, field

from srl.base.rl.config import DummyRLConfig
from srl.rl.memories.replay_buffer import ReplayBufferConfig, RLReplayBuffer


@dataclass
class MyConfig(DummyRLConfig):
    batch_size: int = 2  # batch_sizeを実装する必要あり
    memory: ReplayBufferConfig = field(
        default_factory=lambda: ReplayBufferConfig(
            warmup_size=2,
        )
    )


class MyRemoteMemory(RLReplayBuffer):
    pass


memory = MyRemoteMemory(MyConfig())
memory.add(1)
memory.add(2)
memory.add(3)
memory.add(4)
dat = memory.sample()
print(dat)  # [3, 2]
