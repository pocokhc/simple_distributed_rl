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
        self.prev_rate = 0.0

    def update(self, step: int) -> bool:
        if step >= self.phase_steps:
            rate = 0.0
        else:
            rate = np.cos(self.step_pi * step)
        is_update = self.prev_rate != rate
        self.prev_rate = rate
        return is_update

    def get_rate(self) -> float:
        return self.prev_rate


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
        self.prev_val = 0.0
        self.update(0)

    def update(self, step: int) -> bool:
        if step >= self.phase_steps:
            val = 0.0
        else:
            val = np.cos(self.step_pi * (step % self.cycle_steps))
        is_update = self.prev_val != val
        self.prev_val = val
        return is_update

    def get_rate(self) -> float:
        return self.prev_val
