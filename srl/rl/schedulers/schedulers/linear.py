from dataclasses import dataclass

from .base import BaseScheduler


@dataclass
class Linear(BaseScheduler):
    start_rate: float
    end_rate: float
    phase_steps: int

    def __post_init__(self):
        self.is_up = self.start_rate < self.end_rate
        if self.is_up:
            self.step_rate = (self.end_rate - self.start_rate) / self.phase_steps
        else:
            self.step_rate = (self.start_rate - self.end_rate) / self.phase_steps

    def get_rate(self, step: int) -> float:
        if self.is_up:
            rate = self.start_rate + self.step_rate * step
            if rate > self.end_rate:
                rate = self.end_rate
        else:
            rate = self.start_rate - self.step_rate * step
            if rate < self.end_rate:
                rate = self.end_rate
        return rate
