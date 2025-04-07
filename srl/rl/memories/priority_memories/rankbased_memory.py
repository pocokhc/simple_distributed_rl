import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from .imemory import IPriorityMemory

logger = logging.getLogger(__name__)


@dataclass
class RankBasedMemory(IPriorityMemory):
    capacity: int = 100_000
    #: priorityの反映度、0の場合は完全ランダム、1に近づくほどpriorityによるランダム度になります。
    alpha: float = 0.6
    #: βはISを反映させる割合。ただβは少しずつ増やし、最後に1(完全反映)にします。そのβの初期値です。
    beta_initial: float = 0.4
    #: βを何stepで1にするか
    beta_steps: int = 1_000_000
    dtype: type = np.float32

    def __post_init__(self):
        self.clear()

    def clear(self):
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0

    def length(self) -> int:
        return len(self.buffer)

    def add(self, batch, priority: Optional[float] = None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(batch)
        else:
            self.buffer[self.pos] = batch

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, step: int):
        beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
        if beta > 1:
            beta = 1

        N = len(self.buffer)
        sorted_indices = np.argsort(-self.priorities[:N])
        ranks = np.arange(1, N + 1)
        probs = (1 / ranks) ** self.alpha
        probs /= probs.sum()
        reordered_probs = np.empty_like(probs)
        reordered_probs[sorted_indices] = probs

        sampled_indices = np.random.choice(sorted_indices, size=batch_size, p=probs, replace=False)
        weights = (N * reordered_probs[sampled_indices]) ** (-beta)
        weights = weights / weights.max()
        return [self.buffer[idx] for idx in sampled_indices], weights.astype(self.dtype), sampled_indices

    def update(self, indices: List[Any], priorities: np.ndarray) -> None:
        for idx, td in zip(indices, priorities):
            self.priorities[idx] = td

    def backup(self):
        return [
            self.capacity,
            self.buffer[:],
            self.priorities[:],
            self.pos,
        ]

    def restore(self, data):
        if self.capacity != data[0]:
            logger.warning("Capacity mismatch: expected %d, but got %d", self.capacity, data[0])
        self.buffer = data[1][:]
        self.priorities = data[2][:]
        self.pos = data[3]
