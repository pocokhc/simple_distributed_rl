import bisect
import math
import random
from dataclasses import dataclass
from typing import Any, List

import numpy as np
from srl.base.rl.memory import Memory


def rank_sum(k, a):
    return k * (2 + (k - 1) * a) / 2


def rank_sum_inverse(k, a):
    if a == 0:
        return k
    t = a - 2 + math.sqrt((2 - a) ** 2 + 8 * a * k)
    return t / (2 * a)


class _bisect_wrapper:
    def __init__(self, data, priority):
        self.data = data
        self.priority = priority

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
        self.max_priority = 1

    def add(self, batch, priority=0):
        if priority == 0:
            priority = self.max_priority
        if self.capacity <= len(self.memory):
            # 上限より多い場合は要素を削除
            self.memory.pop(0)

        batch = _bisect_wrapper(batch, priority)
        bisect.insort(self.memory, batch)

    def update(self, indices: List[int], batchs: List[Any], priorities: List[float]) -> None:
        for i in range(len(batchs)):

            batch = _bisect_wrapper(batchs[i], priorities[i])
            bisect.insort(self.memory, batch)

            if self.max_priority < priorities[i]:
                self.max_priority = priorities[i]

    def sample2(self, batch_size, step):
        indices = [0 for _ in range(batch_size)]
        batchs = []
        weights = np.ones(batch_size, dtype="float32")

        # βは最初は低く、学習終わりに1にする。
        beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
        if beta > 1:
            beta = 1

        memory_size = len(self.memory)
        total = rank_sum(memory_size, self.alpha)

        # index_list
        for i in range(batch_size):
            # 合計値をだす
            total2 = rank_sum(len(self.memory), self.alpha)
            r = random.random() * total2
            index = rank_sum_inverse(r, self.alpha)
            index = int(index)  # 整数にする(切り捨て)

            o = self.memory.pop(index)
            batchs.append(o.data)

            # 重点サンプリング
            r1 = rank_sum(index + 1, self.alpha)
            r2 = rank_sum(index, self.alpha)
            prob = (r1 - r2) / total
            weights[i] = (memory_size * prob) ** (-beta)

        # 安定性の理由から最大値で正規化
        weights = weights / weights.max()

        return (indices, batchs, weights)

    def sample(self, batch_size, step):
        indices = [0 for _ in range(batch_size)]
        batchs = []
        weights = np.ones(batch_size, dtype="float32")

        # βは最初は低く、学習終わりに1にする。
        beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
        if beta > 1:
            beta = 1

        # 合計値をだす
        memory_size = len(self.memory)
        total = rank_sum(memory_size, self.alpha)

        # index_list
        index_list = []
        for _ in range(batch_size):

            # index_listにないものを追加
            for _ in range(9999):  # for safety
                r = random.random() * total
                index = rank_sum_inverse(r, self.alpha)
                index = int(index)  # 整数にする(切り捨て)
                if index not in index_list:
                    index_list.append(index)
                    break
        # assert len(index_list) == batch_size

        index_list.sort(reverse=True)
        for i, index in enumerate(index_list):
            o = self.memory.pop(index)  # 後ろから取得するのでindexに変化なし
            batchs.append(o.data)

            # 重点サンプリングを計算 w = (N * pi)
            r1 = rank_sum(index + 1, self.alpha)
            r2 = rank_sum(index, self.alpha)
            prob = (r1 - r2) / total
            weights[i] = (memory_size * prob) ** (-beta)

        # 安定性の理由から最大値で正規化
        weights = weights / weights.max()

        return (indices, batchs, weights)

    def __len__(self):
        return len(self.memory)

    def backup(self):
        return [(d.data, d.priority) for d in self.memory]

    def restore(self, data):
        self.memory = []
        self.max_priority = 1
        for d in data:
            self.add(d[0], d[1])


if __name__ == "__main__":
    pass
