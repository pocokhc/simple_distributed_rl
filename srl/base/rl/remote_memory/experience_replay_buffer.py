import random
from collections import deque
from typing import Any

from srl.base.rl.base import RLRemoteMemory


class ExperienceReplayBuffer(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)

    def init(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def length(self) -> int:
        return len(self.memory)

    def restore(self, data: Any) -> None:
        self.memory = data

    def backup(self):
        return self.memory

    # ---------------------------
    def add(self, batch: Any):
        self.memory.append(batch)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
