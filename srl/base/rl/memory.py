import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class MemoryConfig(ABC):
    @staticmethod
    @abstractmethod
    def getName() -> str:
        raise NotImplementedError()


class Memory(ABC):
    @abstractmethod
    def add(self, exp: Any, priority: float = 0) -> None:
        raise NotImplementedError()

    @abstractmethod
    def update(self, indexes: List[int], batchs: List[Any], priorities: List[float]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, batch_size: int, steps: int) -> Tuple[list, list, list]:
        raise NotImplementedError()
        # return (indexes, batchs, weights)

    @abstractmethod
    def length(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def backup(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def restore(self, data: Any) -> None:
        raise NotImplementedError()

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.backup(), f)

    def load(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        with open(path, "rb") as f:
            self.restore(pickle.load(f))


if __name__ == "__main__":
    pass
