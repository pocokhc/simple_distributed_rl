import logging
import random
from dataclasses import dataclass
from typing import Any, List, Optional, cast

import numpy as np

from .imemory import IPriorityMemory

logger = logging.getLogger(__name__)


class SumTree:
    """Segment Tree, full binary tree
    all nodes: 2N-1
    left     : 2i+1
    right    : 2i+2
    parent   : (i-1)/2
    data -> tree index : j + N - 1
    tree index -> data : i - N

    --- N=5
    data_index, tree_index: priority
    0, 5: 10
    1, 6: 2
    2, 7: 5
    3, 8: 8
    4, 9: 4

    index tree
          0
       1──┴──2
     3─┴─4  5┴6
    7┴8 9┘

    priority tree
         29
      17──┴──12
    13─┴─4  10┴2
    5┴8 4┘
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.write = 0
        self.tree: List[float] = [0 for _ in range(2 * self.capacity - 1)]
        self.data = [None for _ in range(capacity)]

    @staticmethod
    def _propagate(tree, idx: int, change: float):
        while idx != 0:
            parent = (idx - 1) // 2
            tree[parent] += change
            idx = parent

    @staticmethod
    def _retrieve(tree, idx: int, val: float):
        while True:
            left = 2 * idx + 1
            if left >= len(tree):
                return idx
            if val <= tree[left]:
                idx = left
            else:
                idx = left + 1
                val -= tree[left]

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
        self._propagate(self.tree, tree_idx, change)

    def get(self, val):
        idx = self._retrieve(self.tree, 0, float(val))
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

        self.tree.add(cast(float, priority), batch)
        self.size += 1
        if self.size > self.capacity:
            self.size = self.capacity

    def sample(self, batch_size: int, step: int):
        indices = []
        batches = []
        weights = np.empty(batch_size)
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

                if priority == 0:
                    # 多分、並列でtreeで取り出す間に更新が入ると存在しないデータにアクセスしてそう
                    continue

                # 重複を許可しない場合はやり直す
                if (idx in indices) and (not self.has_duplicate):
                    continue
                break

            indices.append(idx)
            batches.append(batch)

            # 重要度サンプリングを計算 w = (N * pi)
            prob = priority / total
            weights[i] = (self.size * prob) ** (-beta)

        # 最大値で正規化
        weights = weights / weights.max()

        return batches, weights, indices

    def update(self, indices: List[Any], priorities: np.ndarray) -> None:
        priorities = (np.abs(priorities) + self.epsilon) ** self.alpha
        for i in range(len(indices)):
            priority = float(priorities[i])
            self.tree.update(indices[i], priority)
            if self.max_priority < priority:
                self.max_priority = priority

    def backup(self):
        return [
            self.capacity,
            self.max_priority,
            self.size,
            self.tree.write,
            self.tree.tree[:],
            self.tree.data[:],
        ]

    def restore(self, data):
        if self.capacity == data[0]:
            self.max_priority = data[1]
            self.size = data[2]
            self.tree.write = data[3]
            self.tree.tree = data[4][:]
            self.tree.data = data[5][:]
        else:
            self.clear()
            capacity = data[0]
            size = data[2]
            tree = data[4]
            tree_data = data[5]
            for i in range(size):
                d = tree_data[i]
                priority = tree[i + capacity - 1]
                self.add(d, priority, _restore_skip=True)
