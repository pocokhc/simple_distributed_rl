import logging
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


class Memory(ABC):
    @staticmethod
    @abstractmethod
    def getName() -> str:
        raise NotImplementedError()

    @abstractmethod
    def init(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def add(self, batch: Any, priority: float = 0) -> None:
        raise NotImplementedError()

    @abstractmethod
    def update(self, indices: List[int], batchs: List[Any], priorities: List[float]) -> None:
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
