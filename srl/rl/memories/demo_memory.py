import logging
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from srl.base.rl.memory import IPriorityMemory, IPriorityMemoryConfig
from srl.rl.memories.config import ReplayMemoryConfig

logger = logging.getLogger(__name__)


@dataclass
class DemoMemory(IPriorityMemory):
    playing: bool = False
    ratio: float = 1.0 / 256.0  # 混ぜる割合
    memory: IPriorityMemoryConfig = field(default_factory=lambda: ReplayMemoryConfig())
    demo_memory: IPriorityMemoryConfig = field(default_factory=lambda: ReplayMemoryConfig())

    def __post_init__(self):
        self._memory = self.memory.create_memory()
        self._demo_memory = self.demo_memory.create_memory()
        self._demo_memory.init()
        self.best_batchs = []
        self.init()

    def init(self) -> None:
        self._memory.init()
        if self.playing:
            self.best_batchs = []

    def add(self, batch: Any, td_error: Optional[float] = None):
        if self.playing:
            self.best_batchs.append(batch)
        else:
            self._memory.add(batch, td_error)

    def sample(self, batch_size: int, step: int) -> Tuple[List[int], List[Any], np.ndarray]:
        self.demo_batch_size = sum([random.random() < self.ratio for _ in range(batch_size)])
        if len(self._demo_memory) < self.demo_batch_size:
            self.demo_batch_size = len(self._demo_memory)
        main_batch_size = batch_size - self.demo_batch_size

        # 比率に基づき batch を作成
        indices = []
        batchs = []
        weights = []
        if self.demo_batch_size > 0:
            (i, b, w) = self._demo_memory.sample(self.demo_batch_size, step)
            indices.extend(i)
            batchs.extend(b)
            weights.extend(w)
        if main_batch_size > 0:
            (i, b, w) = self._memory.sample(main_batch_size, step)
            indices.extend(i)
            batchs.extend(b)
            weights.extend(w)

        return indices, batchs, np.asarray(weights)

    def update(self, indices: List[int], batchs: List[Any], td_errors: np.ndarray) -> None:
        # sample -> update の順番前提
        demo_indices = indices[: self.demo_batch_size]
        demo_batchs = batchs[: self.demo_batch_size]
        demo_td_errors = td_errors[: self.demo_batch_size]
        self._demo_memory.update(demo_indices, demo_batchs, demo_td_errors)

        main_indices = indices[self.demo_batch_size :]
        main_batchs = batchs[self.demo_batch_size :]
        main_td_errors = td_errors[self.demo_batch_size :]
        self._memory.update(main_indices, main_batchs, main_td_errors)

    def __len__(self) -> int:
        return len(self._demo_memory) + len(self._memory)

    def backup(self):
        if self.playing:
            logger.info(f"batch size: {len(self.best_batchs)}")
            return [self.best_batchs]
        else:
            return [
                self.best_batchs,
                self._memory.backup(),
            ]

    def restore(self, data):
        # self.best_batchs = data[0]
        for b in data[0]:
            self._demo_memory.add(b)
        if len(data) == 2:
            self._memory.restore(data[1])
