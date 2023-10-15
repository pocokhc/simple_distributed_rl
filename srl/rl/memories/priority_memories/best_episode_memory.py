import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from .imemory import IPriorityMemory


@dataclass
class BestEpisodeMemory(IPriorityMemory):
    main_memory: IPriorityMemory
    best_memory: IPriorityMemory
    ratio: float = 1.0 / 256.0  # 混ぜる割合
    has_reward_equal: bool = True

    def __post_init__(self):
        self.best_reward = None
        self.best_batchs = []
        self.clear()

    def clear(self) -> None:
        self.main_memory.clear()
        self.episode_batchs = []
        self.episode_reward = 0

    def add(self, batch: Any, priority: Optional[float] = None):
        self.main_memory.add(batch, priority)
        self.episode_batchs.append(batch)

    def on_step(self, reward: float, done: bool) -> None:
        self.episode_reward += reward
        if done:
            _update = False
            if self.best_reward is None:
                _update = True
            else:
                if self.has_reward_equal:
                    if self.best_reward <= self.episode_reward:
                        _update = True
                else:
                    if self.best_reward < self.episode_reward:
                        _update = True
            if _update:
                self.best_reward = self.episode_reward
                self.best_batchs = self.episode_batchs

            self.episode_batchs = []
            self.episode_reward = 0

    def sample(self, step: int, batch_size: int) -> Tuple[List[int], List[Any], np.ndarray]:
        self.best_batch_size = sum([random.random() < self.ratio for _ in range(batch_size)])
        if len(self.best_batchs) < self.best_batch_size:
            self.best_batch_size = len(self.best_batchs)
        main_batch_size = batch_size - self.best_batch_size

        # 比率に基づき batch を作成
        indices = []
        batchs = []
        weights = []
        if self.best_batch_size > 0:
            b = random.sample(self.best_batchs, self.best_batch_size)
            w = [1 for _ in range(self.best_batch_size)]
            i = [0 for _ in range(self.best_batch_size)]
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
        main_indices = indices[self.best_batch_size :]
        main_batchs = batchs[self.best_batch_size :]
        main_priorities = priorities[self.best_batch_size :]
        self.main_memory.update(main_indices, main_batchs, main_priorities)

    def length(self):
        return self.main_memory.length()

    def backup(self):
        return [self.best_batchs, self.best_reward, self.main_memory.backup()]

    def restore(self, data):
        self.episode_batchs = []
        self.episode_reward = 0
        self.best_batchs = data[0]
        self.best_reward = data[1]
        self.main_memory.restore(data[2])
