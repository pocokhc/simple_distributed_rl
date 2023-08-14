from abc import ABC, abstractmethod


class BaseScheduler(ABC):
    @abstractmethod
    def get_rate(self, step: int) -> float:
        raise NotImplementedError()
