from dataclasses import dataclass

from .base import Scheduler


@dataclass
class Linear(Scheduler):
    start_rate: float
    end_rate: float
    phase_steps: int

    def __post_init__(self):
        self.step_rate = (self.start_rate - self.end_rate) / self.phase_steps
        self.rate = self.start_rate

    def update(self, step: int) -> Scheduler:
        if step >= self.phase_steps:
            self.rate = self.end_rate
        else:
            self.rate = self.start_rate - self.step_rate * step
        return self

    def to_float(self) -> float:
        return self.rate
