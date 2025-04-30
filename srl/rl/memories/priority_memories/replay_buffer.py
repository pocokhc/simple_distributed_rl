import random
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from .imemory import IPriorityMemory


@dataclass
class ReplayBuffer(IPriorityMemory):
    capacity: int

    def __post_init__(self):
        self.clear()

    def clear(self):
        self.memory = []
        self.idx = 0

    def length(self) -> int:
        return len(self.memory)

    def add(self, batch: Any, priority: Optional[float] = None):
        if len(self.memory) < self.capacity:
            self.memory.append(batch)
        else:
            self.memory[self.idx] = batch
        self.idx += 1
        if self.idx >= self.capacity:
            self.idx = 0

    def sample(self, batch_size: int, step: int):
        batches = random.sample(self.memory, batch_size)
        return batches, [1.0 for _ in range(batch_size)], []

    def update(self, update_args: List[Any], priorities: np.ndarray) -> None:
        pass

    def backup(self):
        return [
            self.memory[:],
            self.idx,
        ]

    def restore(self, data):
        self.memory = data[0][:]
        self.idx = data[1]
        if len(self.memory) > self.capacity:
            self.idx -= len(self.memory) - self.capacity
            if self.idx < 0:
                self.idx = 0
            self.memory = self.memory[-self.capacity :]
        if self.idx >= self.capacity:
            self.idx = 0
