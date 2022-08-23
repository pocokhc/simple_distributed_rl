import random
from typing import List, Tuple

import numpy as np
from srl.base.define import ContinuousAction, DiscreteAction, RLObservation
from srl.base.env.base import SpaceBase


class DiscreteSpace(SpaceBase[int]):
    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def n(self) -> int:
        return self._n

    def sample(self, invalid_actions: List[int] = []) -> int:
        return random.choice([a for a in range(self.n) if a not in invalid_actions])

    def __eq__(self, o: object) -> bool:
        return self.n == o.n

    def __str__(self) -> str:
        return f"Discrete({self.n})"

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

    # --- observation
    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (1,)

    def observation_discrete_encode(self, val: int) -> RLObservation:
        return np.array([val], dtype=int)

    def observation_continuous_encode(self, val: int) -> RLObservation:
        return np.array([val], dtype=np.float32)
