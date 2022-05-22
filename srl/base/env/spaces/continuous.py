import random
from typing import List, Tuple

import numpy as np
from srl.base.define import ContinuousAction, DiscreteSpaceType, RLObservation
from srl.base.env.spaces.box import BoxSpace


class ContinuousSpace(BoxSpace):
    def __init__(self, low: float = -np.inf, high: float = np.inf) -> None:
        super().__init__((1,), low, high)

    def sample(self, invalid_actions: List[DiscreteSpaceType] = []) -> float:
        return float(super().sample(invalid_actions)[0])

    # --- action discrete
    def get_action_discrete_info(self) -> int:
        return self._n

    def action_discrete_encode(self, val: float) -> int:
        raise NotImplementedError

    def action_discrete_decode(self, val: int) -> float:
        return float(super().action_discrete_decode(val)[0])

    # --- action continuous
    def get_action_continuous_info(self) -> Tuple[int, np.ndarray, np.ndarray]:
        return 1, self.low, self.high

    # def action_continuous_encode(self, val: float) -> ContinuousAction:
    #    return [val]

    def action_continuous_decode(self, val: ContinuousAction) -> float:
        return val[0]

    # --- observation discrete
    def get_observation_discrete_info(self) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray]:
        return (1,), self.low, self.high

    def observation_discrete_encode(self, val: float) -> RLObservation:
        return super().observation_discrete_encode(np.asarray([val]))

    # --- observation continuous
    def get_observation_continuous_info(self) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray]:
        return (1,), self.low, self.high

    def observation_continuous_encode(self, val: float) -> RLObservation:
        return super().observation_continuous_encode(np.asarray([val]))
