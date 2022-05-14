import random
from dataclasses import dataclass
from typing import Any, List

from srl.base.rl.memory import Memory


@dataclass
class ReplayMemory(Memory):

    capacity: int = 100_000

    # no use
    alpha: float = 0
    beta_initial: float = 0
    beta_steps: int = 0

    @staticmethod
    def getName() -> str:
        return "ReplayMemory"

    def __post_init__(self):
        self.init()

    def init(self):
        self.index = 0
        self.memory = []

    def add(self, batch, priority=0):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = batch
        self.index = (self.index + 1) % self.capacity

    def update(self, indices: List[int], batchs: List[Any], priorities: List[float]) -> None:
        pass

    def sample(self, batch_size, step):
        batchs = random.sample(self.memory, batch_size)
        indices = [0 for _ in range(batch_size)]
        weights = [1 for _ in range(batch_size)]
        return (indices, batchs, weights)

    def __len__(self) -> int:
        return len(self.memory)

    def backup(self):
        return self.memory[:]

    def restore(self, data):
        for d in data:
            self.add(d)
