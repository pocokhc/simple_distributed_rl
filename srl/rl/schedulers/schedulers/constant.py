from dataclasses import dataclass

from .base import Scheduler


@dataclass
class Constant(Scheduler):
    rate: float

    def update(self, step: int) -> Scheduler:
        return self

    def get_rate(self) -> float:
        return self.rate
