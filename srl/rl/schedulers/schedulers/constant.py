from dataclasses import dataclass

from .base import BaseScheduler


@dataclass
class Constant(BaseScheduler):
    rate: float

    def get_and_update_rate(self, step: int) -> float:
        return self.rate

    def update(self, step: int) -> bool:
        return False

    def get_rate(self) -> float:
        return self.rate
