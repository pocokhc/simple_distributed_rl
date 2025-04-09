from dataclasses import dataclass, field

import numpy as np

from srl.base.rl.config import DummyRLConfig
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig, RLPriorityReplayBuffer


@dataclass
class MyConfig(DummyRLConfig):
    batch_size: int = 2  # batch_sizeを実装する必要あり
    memory: PriorityReplayBufferConfig = field(
        default_factory=lambda: PriorityReplayBufferConfig(
            warmup_size=2,
        )
    )


class MyRemoteMemory(RLPriorityReplayBuffer):
    pass


config = MyConfig()

# --- select memory
# config.memory.set_replay_buffer()
config.memory.set_proportional()
# config.memory.set_rankbased()
# config.memory.set_rankbased_linear()


# --- run memory
memory = MyRemoteMemory(config)
memory.add(1, priority=1)
memory.add(2, priority=2)
memory.add(3, priority=3)
memory.add(4, priority=4)
batches, weights, update_args = memory.sample()
print(batches)  # [3, 1]
memory.update(update_args, np.array([1, 2]), step=1)
