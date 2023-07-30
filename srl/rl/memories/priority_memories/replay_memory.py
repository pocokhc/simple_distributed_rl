import random
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from .imemory import IPriorityMemory


@dataclass
class ReplayMemory(IPriorityMemory):
    capacity: int = 100_000

    def __post_init__(self):
        self.init()

    def init(self):
        self.memory = []
        self.idx = 0

    def add(self, batch: Any, priority=None):
        if len(self.memory) < self.capacity:
            self.memory.append(batch)
        else:
            self.memory[self.idx] = batch
        self.idx += 1
        if self.idx >= self.capacity:
            self.idx = 0

    def update(self, indices: List[int], batchs: List[Any], td_errors: np.ndarray) -> None:
        pass

    def sample(self, batch_size: int, step: int) -> Tuple[List[int], List[Any], np.ndarray]:
        batchs = random.sample(self.memory, batch_size)
        return [], batchs, np.ones((batch_size,), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.memory)

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
