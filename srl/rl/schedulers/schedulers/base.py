from abc import ABC, abstractmethod


class Scheduler(ABC):
    @abstractmethod
    def update(self, step: int) -> "Scheduler":
        raise NotImplementedError()

    @abstractmethod
    def get_rate(self) -> float:
        raise NotImplementedError()

    def __float__(self) -> float:
        return self.get_rate()

    def to_float(self) -> float:
        return self.get_rate()
