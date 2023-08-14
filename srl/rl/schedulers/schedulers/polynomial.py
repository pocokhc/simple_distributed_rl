from dataclasses import dataclass

from .base import BaseScheduler


@dataclass
class Polynomial(BaseScheduler):
    start_rate: float
    phase_steps: int
    power: float

    def __post_init__(self):
        assert self.start_rate > 0
        assert self.power > 0

    def get_rate(self, step: int) -> float:
        if step >= self.phase_steps:
            return 0.0
        return self.start_rate * (1 - (step / float(self.phase_steps))) ** self.power
