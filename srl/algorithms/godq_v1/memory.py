import logging
from typing import Optional

from srl.base.rl.memory import RLMemory
from srl.rl.memories.priority_replay_buffer import PriorityReplayBuffer
from srl.rl.memories.replay_buffer import ReplayBuffer

from .config import Config

logger = logging.getLogger(__name__)


class Memory(RLMemory[Config]):
    def setup(self) -> None:
        self.q_memory = PriorityReplayBuffer(self.config.memory, self.config.batch_size, self.config.get_dtype("np"))
        self.register_worker_func_custom(self.add_q, self.q_memory.serialize)
        self.register_trainer_recv_func(self.sample_q)
        self.register_trainer_send_func(self.update_q)

        self.diff_memory = ReplayBuffer(
            self.config.batch_size,
            self.config.memory.capacity,
            self.config.memory.warmup_size,
            self.config.memory.compress,
            self.config.memory.compress_level,
        )
        self.register_worker_func_custom(self.add_diff, self.q_memory.serialize)
        self.register_trainer_recv_func(self.sample_diff)

    def length(self):
        return max(self.q_memory.length(), self.diff_memory.length())

    def call_backup(self, **kwargs):
        return [
            self.q_memory.call_backup(),
            self.diff_memory.call_backup(),
        ]

    def call_restore(self, data, **kwargs) -> None:
        self.q_memory.call_restore(data[0])
        self.diff_memory.call_restore(data[1])

    # -------------------------------------------
    def add_q(self, batch, priority: Optional[float], serialized: bool = False):
        self.q_memory.add(batch, priority, serialized=serialized)

    def sample_q(self):
        return self.q_memory.sample()

    def update_q(self, update_args, priorities, step) -> None:
        self.q_memory.update(update_args, priorities, step)

    # -------------------------------------------
    def add_diff(self, batch, serialized: bool = False):
        self.diff_memory.add(batch, serialized=serialized)

    def sample_diff(self):
        return self.diff_memory.sample()
