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
    def sample(self, batch_size: int, step: int) -> Tuple[List[Any], np.ndarray, List[Any]]:
        raise NotImplementedError()  # return (batchs, weights, update_args)

    @abstractmethod
    def update(self, update_args: List[Any], priorities: np.ndarray) -> None:
        raise NotImplementedError()

    @abstractmethod
    def backup(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def restore(self, data: Any) -> None:
        raise NotImplementedError()
