from dataclasses import dataclass, replace
from typing import Any, List, Optional

import numpy as np
from srl.base.rl.memory import Memory


@dataclass
class RankBaseMemoryNaive(Memory):
    # ジップ分布

    capacity: int = 100_000
    alpha: float = 0.6
    beta_initial: float = 0.4
    beta_steps: int = 1_000_000

    @staticmethod
    def getName() -> str:
        return "RankBaseMemoryNaive"

    def __post_init__(self):
        self.init()

    def init(self):
        self.write = 0
        self.size = 0
        self.memory = [None for _ in range(self.capacity)]
        self.max_priority = 1.0

        self.total = 0
        self.props = np.array([])

    def add(self, batch, td_error: Optional[float] = None):
        if td_error is None:
            priority = self.max_priority
        else:
            priority = abs(td_error)

        self.memory[self.write] = [priority, self.write, batch]
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.size < self.capacity:
            self.size += 1
            self.total += 1 / self.size
            self.props = np.append(self.props, 1 / self.size)

    def update(self, indices: List[int], batchs: List[Any], td_errors: np.ndarray) -> None:
        for i in range(len(batchs)):
            priority = abs(td_errors[i])

            self.memory[indices[i]][0] = priority

            if self.max_priority < priority:
                self.max_priority = priority

    def sample(self, batch_size, step):

        # βは最初は低く、学習終わりに1にする。
        beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
        if beta > 1:
            beta = 1

        probs = self.props / self.total

        sorted_memory = sorted(self.memory[: self.size], reverse=True)
        index_list = np.random.choice([i for i in range(self.size)], size=batch_size, replace=False, p=probs)
        batchs = [sorted_memory[i][2] for i in index_list]
        batchs_prob = np.array([probs[i] for i in index_list])
        indices = [sorted_memory[i][1] for i in index_list]

        # 重点サンプリングを計算 w = (N * pi)
        weights = (self.size * batchs_prob) ** (-beta)

        # 最大値で正規化
        weights = weights / weights.max()

        return indices, batchs, weights

    def __len__(self):
        return self.size

    def backup(self):
        return [
            self.write,
            self.size,
            self.max_priority,
            self.memory[:],
        ]

    def restore(self, data):
        self.init()
        self.write = data[0]
        self.size = data[1]
        self.max_priority = data[2]
        self.memory = data[3]
