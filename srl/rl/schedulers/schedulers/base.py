from abc import ABC, abstractmethod


class BaseScheduler(ABC):
    def get_and_update_rate(self, step: int) -> float:
        self.update(step)
        return self.get_rate()

    @abstractmethod
    def update(self, step: int) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_rate(self) -> float:
        raise NotImplementedError()
