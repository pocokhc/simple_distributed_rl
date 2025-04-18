from dataclasses import dataclass

from .base import Scheduler


@dataclass
class Linear(Scheduler):
    start_rate: float
    end_rate: float
    phase_steps: int

    def __post_init__(self):
        self.is_up = self.start_rate < self.end_rate
        if self.is_up:
            self.step_rate = (self.end_rate - self.start_rate) / self.phase_steps
        else:
            self.step_rate = (self.start_rate - self.end_rate) / self.phase_steps
        self.prev_rate = 0.0
        self.update(0)

    def update(self, step: int) -> Scheduler:
        if self.is_up:
            rate = self.start_rate + self.step_rate * step
            if rate > self.end_rate:
                rate = self.end_rate
        else:
            rate = self.start_rate - self.step_rate * step
            if rate < self.end_rate:
                rate = self.end_rate
        self.prev_rate = rate
        return self

    def get_rate(self) -> float:
        return self.prev_rate
