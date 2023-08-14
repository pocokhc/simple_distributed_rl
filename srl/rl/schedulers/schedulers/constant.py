from dataclasses import dataclass

from .base import BaseScheduler


@dataclass
class Constant(BaseScheduler):
    rate: float

    def get_rate(self, step: int) -> float:
        return self.rate
