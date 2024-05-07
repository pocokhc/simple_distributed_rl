import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from .imemory import IPriorityMemory


class SumTree:
    """Segment Tree, full binary tree
    all nodes: 2N-1
    left     : 2i+1
    right    : 2i+2
    parent   : (i-1)/2
    data -> tree index : j + N - 1
    tree index -> data : i - N + 1

    --- N=5
    data_index, tree_index: priority
    0, 4: 10
    1, 5: 2
    2, 6: 5
    3, 7: 8
    4, 8: 4

    index tree
         0
       1─┴─2
      3┴4 5┴6
     7┴8

    priority tree
          29
       22──┴──7
      12┴10  2┴5
     8┴4

    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.write = 0
        self.tree: List[float] = [0 for _ in range(2 * self.capacity - 1)]
        self.data = [None for _ in range(capacity)]

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, val):
        left = 2 * idx + 1

        # 子がなければ該当
        if left >= len(self.tree):
            return idx

        # left が val 以上なら左に移動
        if val <= self.tree[left]:
            return self._retrieve(left, val)
        else:
            # でなければ左の重さを引いて、右に移動
            right = left + 1
            return self._retrieve(right, val - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority: float, data):
        tree_idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(tree_idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, tree_idx: int, priority: float):
        # numpyよりプリミティブ型の方が早い、再帰されるので無視できないレベルで違いが出る
        change = priority - self.tree[tree_idx]

        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, val):
        idx = self._retrieve(0, float(val))
        data_idx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[data_idx])


@dataclass
class ProportionalMemory(IPriorityMemory):
    capacity: int
    #: priorityの反映度、0の場合は完全ランダム、1に近づくほどpriorityによるランダム度になります。
    alpha: float = 0.6
    #: βはISを反映させる割合。ただβは少しずつ増やし、最後に1(完全反映)にします。そのβの初期値です。
    beta_initial: float = 0.4
    #: βを何stepで1にするか
    beta_steps: int = 1_000_000
    #: sample時に重複をきょかするか
    has_duplicate: bool = True
    #: priorityを0にしないための小さい値
    epsilon: float = 0.0001
    dtype: type = np.float32

    def __post_init__(self):
        self.clear()

    def clear(self):
        self.tree = SumTree(self.capacity)
        self.max_priority: float = 1.0
        self.size = 0

    def length(self) -> int:
        return self.size

    def add(self, batch: Any, priority: Optional[float] = None, _restore_skip: bool = False):
        if priority is None:
            priority = self.max_priority
        elif not _restore_skip:
            priority = (abs(priority) + self.epsilon) ** self.alpha

        self.tree.add(priority, batch)
        self.size += 1
        if self.size > self.capacity:
            self.size = self.capacity

    def sample(self, batch_size: int, step: int):
        indices = []
        batchs = []
        weights = np.empty(batch_size, dtype=self.dtype)
        total = self.tree.total()

        # βは最初は低く、学習終わりに1にする
        beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
        if beta > 1:
            beta = 1

        idx = 0
        batch = None
        priority = 0
        for i in range(batch_size):
            for _ in range(9999):  # for safety
                r = random.random() * total
                idx, priority, batch = self.tree.get(r)

                # 重複を許可しない場合はやり直す
                if idx in indices and not self.has_duplicate:
                    continue
                break

            indices.append(idx)
            batchs.append(batch)

            # 重要度サンプリングを計算 w = (N * pi)
            prob = priority / total
            weights[i] = (self.size * prob) ** (-beta)

        # 最大値で正規化
        weights = weights / weights.max()

        return batchs, weights, indices

    def update(self, indices: List[Any], priorities: np.ndarray) -> None:
        priorities = (np.abs(priorities) + self.epsilon) ** self.alpha
        for i in range(len(indices)):
            priority = float(priorities[i])
            self.tree.update(indices[i], priority)
            if self.max_priority < priority:
                self.max_priority = priority

    def backup(self):
        data = []
        for i in range(self.size):
            d = self.tree.data[i]
            priority = self.tree.tree[i + self.capacity - 1]
            data.append([d, priority])
        return data

    def restore(self, data):
        self.clear()
        for d in data:
            self.add(d[0], d[1], _restore_skip=True)
