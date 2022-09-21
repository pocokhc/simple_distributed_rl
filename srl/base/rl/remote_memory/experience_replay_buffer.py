import logging
import random
from typing import Any, List

from srl.base.rl.base import RLRemoteMemory

logger = logging.getLogger(__name__)


class ExperienceReplayBuffer(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)
        self.init(1_000)

    def init(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.idx = 0

    def length(self) -> int:
        return len(self.memory)

    def call_restore(self, data: Any, **kwargs) -> None:
        self.memory = data[0]
        self.idx = data[1]
        if len(self.memory) > self.capacity:
            self.idx -= len(self.memory) - self.capacity
            if self.idx < 0:
                self.idx = 0
            self.memory = self.memory[-self.capacity :]
        if self.idx >= self.capacity:
            self.idx = 0

    def call_backup(self, **kwargs):
        return [
            self.memory,
            self.idx,
        ]

    # ---------------------------
    def add(self, batch: Any):
        if len(self.memory) < self.capacity:
            self.memory.append(batch)
        else:
            self.memory[self.idx] = batch
        self.idx += 1
        if self.idx >= self.capacity:
            self.idx = 0

    def sample(self, batch_size: int) -> List[Any]:
        if len(self.memory) < batch_size:
            logger.warning(f"memory size: {len(self.memory)} < batch size: {batch_size}")
            batch_size = len(self.memory)
        return random.sample(self.memory, batch_size)

    def clear(self) -> None:
        self.memory.clear()
