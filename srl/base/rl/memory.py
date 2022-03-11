import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, List, Tuple


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
    def update(self, indexes: List[int], batchs: List[Any], priorities: List[float]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, batch_size: int, step: int) -> Tuple[list, list, list]:
        raise NotImplementedError()
        # return (indexes, batchs, weights)

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
