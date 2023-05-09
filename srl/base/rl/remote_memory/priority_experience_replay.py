from typing import Any, List, Optional, Tuple

import numpy as np

from srl.base.rl.base import RLRemoteMemory
from srl.base.rl.memory import IPriorityMemoryConfig


class PriorityExperienceReplay(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)

    def init(self, config: IPriorityMemoryConfig):
        self.memory = config.create_memory()

    def length(self) -> int:
        return len(self.memory)

    def call_restore(self, data: Any, **kwargs) -> None:
        self.memory.restore(data)

    def call_backup(self, **kwargs):
        return self.memory.backup()

    # ---------------------------

    def add(self, batch: Any, td_error: Optional[float] = None):
        self.memory.add(batch, td_error)

    def sample(self, batch_size: int, step: int) -> Tuple[list, list, list]:
        return self.memory.sample(batch_size, step)

    def update(self, indices: List[int], batchs: List[Any], td_errors: np.ndarray) -> None:
        self.memory.update(indices, batchs, td_errors)

    def on_step(self, reward: float, done: bool) -> None:
        self.memory.on_step(reward, done)
