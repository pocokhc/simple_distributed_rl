from typing import Any, List, Tuple

import numpy as np

from srl.base.define import ContinuousAction, DiscreteSpaceType, RLObservation

from .box import BoxSpace


class ContinuousSpace(BoxSpace):
    def __init__(self, low: float = -np.inf, high: float = np.inf) -> None:
        super().__init__((1,), low, high)

    def sample(self, invalid_actions: List[DiscreteSpaceType] = []) -> float:
        return float(super().sample(invalid_actions)[0])

    def convert(self, val: Any) -> float:
        if isinstance(val, list):
            return float(val[0])
        elif isinstance(val, tuple):
            return float(val[0])
        return float(val)

    def __str__(self) -> str:
        return f"Continuous({self.low} - {self.high})"

    def check_val(self, val: Any) -> bool:
        if not isinstance(val, float):
            return False
        if val < self.low:
            return False
        if val > self.high:
            return False
        return True

    def get_default(self) -> float:
        return 0.0

    # --- action discrete
    def get_action_discrete_info(self) -> int:
        return self._n

    def action_discrete_encode(self, val: float) -> int:
        return super().action_discrete_encode(val)

    def action_discrete_decode(self, val: int) -> float:
        return float(super().action_discrete_decode(val)[0])

    # --- action continuous
    def get_action_continuous_info(self) -> Tuple[int, np.ndarray, np.ndarray]:
        return 1, self.low, self.high

    # def action_continuous_encode(self, val: float) -> ContinuousAction:
    #    return [val]

    def action_continuous_decode(self, val: ContinuousAction) -> float:
        return val[0]

    # --- observation
    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (1,)

    def observation_discrete_encode(self, val: float) -> RLObservation:
        return super().observation_discrete_encode(np.asarray([val]))

    def observation_continuous_encode(self, val: float) -> RLObservation:
        return super().observation_continuous_encode(np.asarray([val]))
