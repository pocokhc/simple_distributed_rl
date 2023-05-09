from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np


class IPriorityMemory(ABC):
    @abstractmethod
    def init(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def add(self, batch: Any, td_error: Optional[float] = None) -> None:
        raise NotImplementedError()

    @abstractmethod
    def update(self, indices: List[int], batchs: List[Any], td_errors: np.ndarray) -> None:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, batch_size: int, step: int) -> Tuple[list, list, list]:
        raise NotImplementedError()  # return (indices, batchs, weights)

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def backup(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def restore(self, data: Any) -> None:
        raise NotImplementedError()

    def clear(self) -> None:
        self.init()

    def on_step(self, reward: float, done: bool):
        pass


@dataclass
class IPriorityMemoryConfig(ABC):
    @abstractmethod
    def create_memory() -> IPriorityMemory:
        raise NotImplementedError()

    @abstractmethod
    def get_capacity(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def is_replay_memory(self) -> bool:
        raise NotImplementedError()
