from dataclasses import dataclass

from .base import Scheduler


@dataclass
class Polynomial(Scheduler):
    start_rate: float
    phase_steps: int
    power: float

    def __post_init__(self):
        assert self.start_rate > 0
        assert self.power > 0
        self.prev_rate = 0.0
        self.update(0)

    def update(self, step: int) -> Scheduler:
        if step >= self.phase_steps:
            rate = 0.0
        else:
            rate = self.start_rate * (1 - (step / float(self.phase_steps))) ** self.power
        self.prev_rate = rate
        return self

    def get_rate(self) -> float:
        return self.prev_rate
