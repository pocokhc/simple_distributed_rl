import logging
from typing import Optional

from srl.base.rl.memory import RLMemory
from srl.rl.memories.priority_replay_buffer import PriorityReplayBuffer

from .config import Config

logger = logging.getLogger(__name__)


class Memory(RLMemory[Config]):
    def setup(self) -> None:
        self.q_memory = PriorityReplayBuffer(self.config.memory, self.config.batch_size, self.config.get_dtype("np"))
        self.register_worker_func(self.add_q, self.q_memory.serialize)
        self.register_trainer_recv_func(self.sample_q)
        self.register_trainer_send_func(self.update_q)

    def length(self):
        return self.q_memory.length()

    def call_backup(self, **kwargs):
        return [
            self.q_memory.call_backup(),
        ]

    def call_restore(self, data, **kwargs) -> None:
        self.q_memory.call_restore(data[0])

    # -------------------------------------------
    def add_q(self, batch, priority: Optional[float], serialized: bool = False):
        self.q_memory.add(batch, priority, serialized=serialized)

    def sample_q(self):
        return self.q_memory.sample()

    def update_q(self, update_args, priorities, step) -> None:
        self.q_memory.update(update_args, priorities, step)
