import bisect
import math
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
        return "RankBaseMemory"


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


class RankBaseMemory(Memory):
    def __init__(self, config: Config):
        self.capacity = config.capacity
        self.alpha = config.alpha
        self.beta_initial = config.beta_initial
        self.beta_steps = config.beta_steps

        self.buffer = []
        self.max_priority = 1

    def add(self, exp, priority=0):
        if priority == 0:
            priority = self.max_priority
        if self.capacity <= len(self.buffer):
            # 上限より多い場合は要素を削除
            self.buffer.pop(0)

        exp = _bisect_wrapper(exp, priority)
        bisect.insort(self.buffer, exp)

    def update(self, indexes: List[int], batchs: List[Any], priorities: List[float]) -> None:
        for i in range(len(batchs)):

            exp = _bisect_wrapper(batchs[i], priorities[i])
            bisect.insort(self.buffer, exp)

            if self.max_priority < priorities[i]:
                self.max_priority = priorities[i]

    def sample2(self, batch_size, step):
        indexes = [0 for _ in range(batch_size)]
        batchs = []
        weights = np.ones(batch_size, dtype="float32")

        # βは最初は低く、学習終わりに1にする。
        beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
        if beta > 1:
            beta = 1

        buffer_size = len(self.buffer)
        total = rank_sum(buffer_size, self.alpha)

        # index_list
        for i in range(batch_size):
            # 合計値をだす
            total2 = rank_sum(len(self.buffer), self.alpha)
            r = random.random() * total2
            index = rank_sum_inverse(r, self.alpha)
            index = int(index)  # 整数にする(切り捨て)

            o = self.buffer.pop(index)
            batchs.append(o.data)

            # 重点サンプリング
            r1 = rank_sum(index + 1, self.alpha)
            r2 = rank_sum(index, self.alpha)
            prob = (r1 - r2) / total
            weights[i] = (buffer_size * prob) ** (-beta)

        # 安定性の理由から最大値で正規化
        weights = weights / weights.max()

        return (indexes, batchs, weights)

    def sample(self, batch_size, step):
        indexes = [0 for _ in range(batch_size)]
        batchs = []
        weights = np.ones(batch_size, dtype="float32")

        # βは最初は低く、学習終わりに1にする。
        beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
        if beta > 1:
            beta = 1

        # 合計値をだす
        buffer_size = len(self.buffer)
        total = rank_sum(buffer_size, self.alpha)

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
            o = self.buffer.pop(index)  # 後ろから取得するのでindexに変化なし
            batchs.append(o.data)

            # 重点サンプリングを計算 w = (N * pi)
            r1 = rank_sum(index + 1, self.alpha)
            r2 = rank_sum(index, self.alpha)
            prob = (r1 - r2) / total
            weights[i] = (buffer_size * prob) ** (-beta)

        # 安定性の理由から最大値で正規化
        weights = weights / weights.max()

        return (indexes, batchs, weights)

    def length(self):
        return len(self.buffer)

    def backup(self):
        return [(d.data, d.priority) for d in self.buffer]

    def restore(self, data):
        self.buffer = []
        self.max_priority = 1
        for d in data:
            self.add(d[0], d[1])


register(Config, RankBaseMemory)

if __name__ == "__main__":
    pass
