import random
from typing import List, Tuple

import numpy as np
from srl.base.define import ContinuousAction, DiscreteAction, DiscreteSpaceType, RLObservation
from srl.base.env.base import SpaceBase


class DiscreteSpace(SpaceBase):
    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def n(self) -> int:
        return self._n

    def sample(self, invalid_actions: List[DiscreteSpaceType] = []) -> int:
        return random.choice([a for a in range(self.n) if a not in invalid_actions])

    # --- action discrete
    def get_action_discrete_info(self) -> int:
        return self.n

    def action_discrete_encode(self, val: int) -> DiscreteAction:
        return val

    def action_discrete_decode(self, val: DiscreteAction) -> int:
        return val

    # --- action continuous
    def get_action_continuous_info(self) -> Tuple[int, np.ndarray, np.ndarray]:
        return 1, np.array([0]), np.array([self.n - 1])

    # def action_continuous_encode(self, val: int) -> ContinuousAction:
    #    return [float(val)]

    def action_continuous_decode(self, val: ContinuousAction) -> int:
        return int(np.round(val[0]))

    # --- observation discrete
    def get_observation_discrete_info(self) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray]:
        return (1,), np.array([0]), np.array([self.n - 1])

    def observation_discrete_encode(self, val: int) -> RLObservation:
        return np.array([val], dtype=int)

    # --- observation continuous
    def get_observation_continuous_info(self) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray]:
        return (1,), np.array([0]), np.array([self.n - 1])

    def observation_continuous_encode(self, val: int) -> RLObservation:
        return np.array([val], dtype=float)
