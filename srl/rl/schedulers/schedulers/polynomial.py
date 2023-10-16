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
        self.prev_rate = 0.0
        self.update(0)

    def update(self, step: int) -> bool:
        if step >= self.phase_steps:
            rate = 0.0
        else:
            rate = self.start_rate * (1 - (step / float(self.phase_steps))) ** self.power
        is_update = self.prev_rate != rate
        self.prev_rate = rate
        return is_update

    def get_rate(self) -> float:
        return self.prev_rate
