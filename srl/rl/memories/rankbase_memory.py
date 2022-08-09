import bisect
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
from srl.base.rl.memory import Memory


class _bisect_wrapper:
    def __init__(self, priority, batch):
        self.priority = priority
        self.batch = batch

    def __lt__(self, o):  # a<b
        return self.priority < o.priority


@dataclass
class RankBaseMemory(Memory):

    capacity: int = 100_000
    alpha: float = 0.6
    beta_initial: float = 0.4
    beta_steps: int = 1_000_000

    @staticmethod
    def getName() -> str:
        return "RankBaseMemory"

    def __post_init__(self):
        self.init()

    def init(self):
        self.memory = []
        self.max_priority = 1.0
        self.total = 0.0
        self.probs = np.array([])
        self.is_full = False

        self.total_probs = []

    def add(self, batch, td_error: Optional[float] = None):
        if td_error is None:
            priority = self.max_priority
        else:
            priority = float(abs(td_error))
            if self.max_priority < priority:
                self.max_priority = priority

        bisect.insort(self.memory, _bisect_wrapper(priority, batch))

        if not self.is_full:
            p = (1 / len(self.memory)) ** self.alpha
            self.total += p
            self.probs = np.append(self.probs, p)  # 降順
            self.total_probs.append(self.total)

        if len(self.memory) == self.capacity:
            self.is_full = True
            self.probs /= self.total

        elif len(self.memory) > self.capacity:
            self.memory.pop(0)

    def update(self, indices: List[int], batchs: List[Any], td_errors: np.ndarray) -> None:
        for i in range(len(batchs)):
            priority = float(abs(td_errors[i]))
            if self.max_priority < priority:
                self.max_priority = priority
            bisect.insort(self.memory, _bisect_wrapper(priority, batchs[i]))

    def sample(self, batch_size, step):

        # βは最初は低く、学習終わりに1にする。
        beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
        if beta > 1:
            beta = 1

        N = len(self.memory)

        if self.is_full:
            probs = self.probs
        else:
            probs = self.probs / self.total

        index_list = np.random.choice([i for i in range(N)], size=batch_size, replace=False, p=probs)
        index_list.sort()

        batchs = []
        weights = np.ones(batch_size, dtype=np.float32)

        for i, index in enumerate(index_list):
            memory_idx = N - 1 - index
            o = self.memory.pop(memory_idx)  # 後ろから取得してindexの変化を防ぐ

            batchs.append(o.batch)
            prob = probs[index]
            weights[i] = (N * prob) ** (-beta)

        # 最大値で正規化
        weights = weights / weights.max()

        return None, batchs, weights

    def __len__(self):
        return len(self.memory)

    def backup(self):
        return [
            [(o.priority, o.batch) for o in self.memory],
            self.max_priority,
            self.total,
            self.probs.tolist(),
            self.is_full,
        ]

    def restore(self, data):
        self.init()
        for d in data[0]:
            self.memory.append(_bisect_wrapper(d[0], d[1]))
        self.max_priority = data[1]
        self.total = data[2]
        self.probs = np.array(data[3])
        self.is_full = data[4]
