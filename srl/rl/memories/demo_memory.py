import logging
import random
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from srl.base.rl.memory import RLMemory

logger = logging.getLogger(__name__)

# TODO


@dataclass
class DemoMemory(RLMemory):
    main_memory: RLMemory
    demo_memory: RLMemory
    ratio: float = 1.0 / 256.0  # 混ぜる割合

    def add(self, *args) -> None:
        raise NotImplementedError("not implement")

    def length(self) -> int:
        return self.demo_memory.length() + self.main_memory.length()

    def sample(self, batch_size: int, step: int) -> Tuple[List[int], List[Any], np.ndarray]:
        if self.demo_memory.length() == 0:
            self.demo_batch_size = 0
            main_batch_size = batch_size
        elif self.main_memory.length() == 0:
            self.demo_batch_size = batch_size
            main_batch_size = 0
        else:
            self.demo_batch_size = sum([random.random() < self.ratio for _ in range(batch_size)])
            if self.demo_memory.length() < self.demo_batch_size:
                self.demo_batch_size = self.demo_memory.length()
            main_batch_size = batch_size - self.demo_batch_size

        # 比率に基づき batch を作成
        indices = []
        batchs = []
        weights = []
        if self.demo_batch_size > 0:
            (i, b, w) = self.demo_memory.sample(step, self.demo_batch_size)
            indices.extend(i)
            batchs.extend(b)
            weights.extend(w)
        if main_batch_size > 0:
            (i, b, w) = self.main_memory.sample(step, main_batch_size)
            indices.extend(i)
            batchs.extend(b)
            weights.extend(w)

        return indices, batchs, np.asarray(weights, dtype=np.float32)

    def update(self, indices: List[int], batchs: List[Any], priorities: np.ndarray) -> None:
        # sample -> update の順番前提
        demo_indices = indices[: self.demo_batch_size]
        demo_batchs = batchs[: self.demo_batch_size]
        demo_priorities = priorities[: self.demo_batch_size]
        self.demo_memory.update((demo_indices, demo_batchs, demo_priorities))

        main_indices = indices[self.demo_batch_size :]
        main_batchs = batchs[self.demo_batch_size :]
        main_priorities = priorities[self.demo_batch_size :]
        self.main_memory.update((main_indices, main_batchs, main_priorities))

    def call_backup(self):
        return [
            self.demo_memory.backup(),
            self.main_memory.backup(),
        ]

    def call_restore(self, data):
        self.demo_memory.restore(data[0])
        if len(data) == 2:
            self.main_memory.restore(data[1])
