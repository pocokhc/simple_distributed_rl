import logging
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from .imemory import IPriorityMemory

logger = logging.getLogger(__name__)


@dataclass
class DemoMemory(IPriorityMemory):
    main_memory: IPriorityMemory
    demo_memory: IPriorityMemory
    playing: bool = False
    ratio: float = 1.0 / 256.0  # 混ぜる割合

    def __post_init__(self):
        self.init()

    def init(self) -> None:
        self.main_memory.init()
        if self.playing:
            self.demo_memory.init()

    def add(self, batch: Any, td_error: Optional[float] = None):
        if self.playing:
            self.demo_memory.add(batch, td_error)
        else:
            self.main_memory.add(batch, td_error)

    def sample(self, batch_size: int, step: int) -> Tuple[List[int], List[Any], np.ndarray]:
        if len(self.demo_memory) == 0:
            self.demo_batch_size = 0
            main_batch_size = batch_size
        elif len(self.main_memory) == 0:
            self.demo_batch_size = batch_size
            main_batch_size = 0
        else:
            self.demo_batch_size = sum([random.random() < self.ratio for _ in range(batch_size)])
            if len(self.demo_memory) < self.demo_batch_size:
                self.demo_batch_size = len(self.demo_memory)
            main_batch_size = batch_size - self.demo_batch_size

        # 比率に基づき batch を作成
        indices = []
        batchs = []
        weights = []
        if self.demo_batch_size > 0:
            (i, b, w) = self.demo_memory.sample(self.demo_batch_size, step)
            indices.extend(i)
            batchs.extend(b)
            weights.extend(w)
        if main_batch_size > 0:
            (i, b, w) = self.main_memory.sample(main_batch_size, step)
            indices.extend(i)
            batchs.extend(b)
            weights.extend(w)

        return indices, batchs, np.asarray(weights, dtype=np.float32)

    def update(self, indices: List[int], batchs: List[Any], td_errors: np.ndarray) -> None:
        # sample -> update の順番前提
        demo_indices = indices[: self.demo_batch_size]
        demo_batchs = batchs[: self.demo_batch_size]
        demo_td_errors = td_errors[: self.demo_batch_size]
        self.demo_memory.update(demo_indices, demo_batchs, demo_td_errors)

        main_indices = indices[self.demo_batch_size :]
        main_batchs = batchs[self.demo_batch_size :]
        main_td_errors = td_errors[self.demo_batch_size :]
        self.main_memory.update(main_indices, main_batchs, main_td_errors)

    def __len__(self) -> int:
        return len(self.demo_memory) + len(self.main_memory)

    def backup(self):
        if self.playing:
            return [self.demo_memory.backup()]
        else:
            return [
                self.demo_memory.backup(),
                self.main_memory.backup(),
            ]

    def restore(self, data):
        self.demo_memory.restore(data[0])
        if len(data) == 2:
            self.main_memory.restore(data[1])
