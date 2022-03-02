import random
from dataclasses import dataclass
from typing import Any, List

import numpy as np
from srl.base.rl.memory import Memory, MemoryConfig
from srl.rl.memory.registory import register


@dataclass
class Config(MemoryConfig):
    capacity: int = 100_000
    alpha: float = 0.6
    beta_initial: float = 0.4
    beta_steps: int = 1_000_000

    @staticmethod
    def getName() -> str:
        return "ProportionalMemory"


class SumTree:
    """
    copy from https://github.com/jaromiru/AI-blog/blob/5aa9f0b/SumTree.py
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.write = 0
        self.tree = [0 for _ in range(2 * capacity - 1)]
        self.data = [None for _ in range(capacity)]

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class ProportionalMemory(Memory):
    def __init__(self, config: Config):
        self.capacity = config.capacity
        self.alpha = config.alpha
        self.beta_initial = config.beta_initial
        self.beta_steps = config.beta_steps

        self.tree = SumTree(self.capacity)
        self.max_priority = 1
        self.size = 0

    def add(self, exp, priority=0, _alpha_skip=False):
        if priority == 0:
            priority = self.max_priority
        if not _alpha_skip:
            priority = priority**self.alpha
        self.tree.add(priority, exp)
        self.size += 1
        if self.size > self.capacity:
            self.size = self.capacity

    def update(self, indexes: List[int], batchs: List[Any], priorities: List[float]) -> None:
        for i in range(len(batchs)):
            priority = priorities[i] ** self.alpha
            self.tree.update(indexes[i], priority)

            if self.max_priority < priority:
                self.max_priority = priority

    def sample(self, batch_size, step):
        indexes = []
        batchs = []
        weights = np.empty(batch_size, dtype="float32")

        # βは最初は低く、学習終わりに1にする
        beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
        if beta > 1:
            beta = 1

        total = self.tree.total()
        for i in range(batch_size):
            idx = None
            experience = None
            priority = 0

            # indexesにないものを追加
            for _ in range(9999):  # for safety
                r = random.random() * total
                (idx, priority, experience) = self.tree.get(r)
                if idx not in indexes:
                    break

            indexes.append(idx)
            batchs.append(experience)

            # 重要度サンプリングを計算 w = (N * pi)
            prob = priority / total
            weights[i] = (self.size * prob) ** (-beta)

        # 安定性の理由から最大値で正規化
        weights = weights / weights.max()

        return (indexes, batchs, weights)

    def length(self):
        return self.size

    def backup(self):
        data = []
        for i in range(self.size):
            d = self.tree.data[i]
            p = self.tree.tree[i + self.capacity - 1]
            data.append([d, p])

        return data

    def restore(self, data):
        self.tree = SumTree(self.capacity)
        self.size = 0

        for d in data:
            self.add(d[0], d[1], _alpha_skip=True)


register(Config, ProportionalMemory)

if __name__ == "__main__":
    pass
