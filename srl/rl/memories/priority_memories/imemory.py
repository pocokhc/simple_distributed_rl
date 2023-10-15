from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import numpy as np


class IPriorityMemory(ABC):
    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def length(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def add(self, batch: Any, priority: Optional[float] = None) -> None:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, step: int, batch_size: int) -> Tuple[List[int], List[Any], np.ndarray]:
        raise NotImplementedError()  # return (indices, batchs, weights)

    @abstractmethod
    def update(self, indices: List[int], batchs: List[Any], priorities: np.ndarray) -> None:
        raise NotImplementedError()

    @abstractmethod
    def backup(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def restore(self, data: Any) -> None:
        raise NotImplementedError()
