from dataclasses import dataclass

import numpy as np

from .base import Scheduler


@dataclass
class Cosine(Scheduler):
    start_rate: float
    end_rate: float
    phase_steps: int

    def __post_init__(self):
        self.step_pi = (np.pi / 2) / self.phase_steps
        self.rate = self.start_rate

    def update(self, step: int) -> Scheduler:
        if step >= self.phase_steps:
            self.rate = self.end_rate
        else:
            rate = np.cos(self.step_pi * step)
            self.rate = (self.start_rate - self.end_rate) * rate + self.end_rate
        return self

    def to_float(self) -> float:
        return self.rate


@dataclass
class CosineWithHardRestarts(Scheduler):
    start_rate: float
    end_rate: float
    phase_steps: int
    num_cycles: int

    def __post_init__(self):
        assert self.num_cycles > 0
        self.cycle_steps = int(self.phase_steps / self.num_cycles)
        self.step_pi = (np.pi / 2) / self.cycle_steps
        self.rate = self.start_rate

    def update(self, step: int) -> Scheduler:
        if step >= self.phase_steps:
            self.rate = self.end_rate
        else:
            rate = np.cos(self.step_pi * (step % self.cycle_steps))
            self.rate = (self.start_rate - self.end_rate) * rate + self.end_rate
        return self

    def to_float(self) -> float:
        return self.rate
