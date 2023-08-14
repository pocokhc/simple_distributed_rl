from dataclasses import dataclass

import numpy as np

from .base import BaseScheduler


@dataclass
class Cosine(BaseScheduler):
    start_rate: float
    phase_steps: int

    def __post_init__(self):
        assert self.start_rate > 0
        self.step_pi = (np.pi / 2) / self.phase_steps

    def get_rate(self, step: int) -> float:
        if step >= self.phase_steps:
            return 0.0
        return np.cos(self.step_pi * step)


@dataclass
class CosineWithHardRestarts(BaseScheduler):
    start_rate: float
    phase_steps: int
    num_cycles: int

    def __post_init__(self):
        assert self.start_rate > 0
        assert self.num_cycles > 0
        self.cycle_steps = int(self.phase_steps / self.num_cycles)
        self.step_pi = (np.pi / 2) / self.cycle_steps

    def get_rate(self, step: int) -> float:
        if step >= self.phase_steps:
            return 0.0
        return np.cos(self.step_pi * (step % self.cycle_steps))
