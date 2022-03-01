import random
from dataclasses import dataclass
from typing import Any

from srl.base.rl.memory import Memory, MemoryConfig
from srl.rl.memory.registory import register


@dataclass
class Config(MemoryConfig):
    capacity: int = 100_000

    @staticmethod
    def getName() -> str:
        return "ReplayMemory"


class ReplayMemory(Memory):
    def __init__(self, config: Config):
        self.capacity = config.capacity

        self.index = 0
        self.buffer = []

    def add(self, exp, priority=0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = exp
        self.index = (self.index + 1) % self.capacity

    def update(self, indexes: list[int], batchs: list[Any], priorities: list[float]) -> None:
        pass

    def sample(self, batch_size, step):
        batchs = random.sample(self.buffer, batch_size)
        indexes = [0 for _ in range(batch_size)]
        weights = [1 for _ in range(batch_size)]
        return (indexes, batchs, weights)

    def length(self) -> int:
        return len(self.buffer)

    def backup(self):
        return self.buffer[:]

    def restore(self, data):
        for d in data:
            self.add(d)


register(Config, ReplayMemory)

if __name__ == "__main__":
    pass
