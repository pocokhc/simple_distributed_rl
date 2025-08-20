from dataclasses import dataclass

from .base import Scheduler


@dataclass
class Polynomial(Scheduler):
    start_rate: float
    end_rate: float
    phase_steps: int
    power: float

    def __post_init__(self):
        assert self.power >= 0
        self.rate = self.start_rate
        self.update(0)

    def update(self, step: int) -> Scheduler:
        if step >= self.phase_steps:
            self.rate = self.end_rate
        else:
            rate = (1 - (step / float(self.phase_steps))) ** self.power
            self.rate = (self.start_rate - self.end_rate) * rate + self.end_rate
        return self

    def to_float(self) -> float:
        return self.rate
