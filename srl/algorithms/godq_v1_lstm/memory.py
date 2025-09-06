import logging

from srl.base.rl.memory import RLMemory
from srl.rl.memories.episode_replay_buffer import EpisodeReplayBuffer

from .config import Config

logger = logging.getLogger(__name__)


class Memory(RLMemory[Config]):
    def setup(self) -> None:
        self.q_memory = EpisodeReplayBuffer(
            self.config.memory,
            self.config.batch_size,
            suffix_size=self.config.batch_length,
            sequential_stride=self.config.batch_length + 1,
        )
        self.register_worker_func(self.add_steps, self.q_memory.serialize)
        self.register_trainer_recv_func(self.sample_sequential)
        self.n = 0

    def length(self):
        return self.q_memory.length()

    def call_backup(self, **kwargs):
        return [
            self.q_memory.call_backup(),
        ]

    def call_restore(self, data, **kwargs) -> None:
        self.q_memory.call_restore(data[0])

    # -------------------------------------------
    def add_steps(self, steps: list, size: int = 0, serialized: bool = False):
        self.q_memory.add(steps, size, serialized=serialized)

    def sample_sequential(self):
        self.n += 1
        if self.n % 2 == 0:
            batch_size = self.config.batch_size
            batch_length = self.config.batch_length + 1
            sequential_stride = self.config.batch_length + 1
        else:
            batch_size = self.config.batch_size // 2
            batch_length = self.config.batch_length + 2
            sequential_stride = self.config.batch_length + 2
        return self.q_memory.sample_sequential(
            batch_size=batch_size,
            batch_length=batch_length,
            sequential_stride=sequential_stride,
        )
