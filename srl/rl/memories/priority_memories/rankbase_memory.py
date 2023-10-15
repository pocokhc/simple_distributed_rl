import bisect
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from .imemory import IPriorityMemory


class _bisect_wrapper:
    def __init__(self, priority, batch):
        self.priority = priority
        self.batch = batch

    def __lt__(self, o):  # a<b
        return self.priority < o.priority


@dataclass
class RankBaseMemory(IPriorityMemory):
    capacity: int = 100_000
    #: priorityの反映度、0の場合は完全ランダム、1に近づくほどpriorityによるランダム度になります。
    alpha: float = 0.6
    #: βはISを反映させる割合。ただβは少しずつ増やし、最後に1(完全反映)にします。そのβの初期値です。
    beta_initial: float = 0.4
    #: βを何stepで1にするか
    beta_steps: int = 1_000_000

    def __post_init__(self):
        self.clear()

    def clear(self):
        self.memory = []
        self.max_priority: float = 1.0
        self.total = 0.0
        self.probs = np.array([])
        self.is_full = False
        self.total_probs = []

    def length(self) -> int:
        return len(self.memory)

    def add(self, batch, priority: Optional[float] = None):
        if priority is None:
            priority = self.max_priority
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

    def sample(self, batch_size: int, step: int) -> Tuple[List[int], List[Any], np.ndarray]:
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

        return [], batchs, weights

    def update(self, indices: List[int], batchs: List[Any], priorities: np.ndarray) -> None:
        for i in range(len(batchs)):
            priority = float(priorities[i])
            if self.max_priority < priority:
                self.max_priority = priority
            bisect.insort(self.memory, _bisect_wrapper(priority, batchs[i]))

    def backup(self):
        return [
            [(o.priority, o.batch) for o in self.memory],
            self.max_priority,
            self.total,
            self.probs.tolist(),
            self.is_full,
        ]

    def restore(self, data):
        self.clear()
        for d in data[0]:
            self.memory.append(_bisect_wrapper(d[0], d[1]))
        self.max_priority = data[1]
        self.total = data[2]
        self.probs = np.array(data[3])
        self.is_full = data[4]
