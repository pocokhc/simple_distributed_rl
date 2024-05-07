import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from .imemory import IPriorityMemory


@dataclass
class ReplayMemory(IPriorityMemory):
    capacity: int
    dtype: type = np.float32

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
        batchs = random.sample(self.memory, batch_size)
        return batchs, np.ones((batch_size,), dtype=self.dtype), []

    def update(self, update_args: List[Any], priorities: np.ndarray) -> None:
        pass

    def backup(self):
        return [
            self.memory,
            self.idx,
        ]

    def restore(self, data):
        self.memory = data[0]
        self.idx = data[1]
        if len(self.memory) > self.capacity:
            self.idx -= len(self.memory) - self.capacity
            if self.idx < 0:
                self.idx = 0
            self.memory = self.memory[-self.capacity :]
        if self.idx >= self.capacity:
            self.idx = 0
