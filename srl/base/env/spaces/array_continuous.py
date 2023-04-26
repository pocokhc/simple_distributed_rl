from typing import Any, List, Tuple, Union

import numpy as np

from srl.base.define import ContinuousAction, DiscreteAction, DiscreteSpaceType, RLObservation

from .box import BoxSpace


class ArrayContinuousSpace(BoxSpace):
    def __init__(
        self,
        size: int,
        low: Union[float, np.ndarray] = -np.inf,
        high: Union[float, np.ndarray] = np.inf,
    ) -> None:
        self._size = size
        super().__init__((self.size,), low, high)

    @property
    def size(self):
        return self._size

    def sample(self, invalid_actions: List[DiscreteSpaceType] = []) -> List[float]:
        return super().sample(invalid_actions).tolist()

    def convert(self, val: Any) -> List[float]:
        if isinstance(val, list):
            return [float(v) for v in val]
        elif isinstance(val, tuple):
            return [float(v) for v in val]
        elif isinstance(val, np.ndarray):
            return val.tolist()
        return [float(val) for _ in range(self.size)]

    def __str__(self) -> str:
        return f"ArrayContinuous({self.size}, range[{np.min(self.low)}, {np.max(self.high)}])"

    def check_val(self, val: Any) -> bool:
        if not isinstance(val, list):
            return False
        if len(val) != self.size:
            return False
        for i in range(self.size):
            if not isinstance(val[i], float):
                return False
            if val[i] < self.low[i]:
                return False
            if val[i] > self.high[i]:
                return False
        return True

    def get_default(self) -> List[float]:
        return [0.0 for _ in range(self.size)]

    # --- action discrete
    def get_action_discrete_info(self) -> int:
        return self._n

    def action_discrete_encode(self, val: List[float]) -> DiscreteAction:
        return super().action_discrete_encode(val)

    def action_discrete_decode(self, val: DiscreteAction) -> List[float]:
        return super().action_discrete_decode(val).tolist()

    # --- action continuous
    def get_action_continuous_info(self) -> Tuple[int, np.ndarray, np.ndarray]:
        return super().get_action_continuous_info()

    def action_continuous_decode(self, val: ContinuousAction) -> List[float]:
        return val

    # --- observation
    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return super().observation_shape

    def observation_discrete_encode(self, val: List[float]) -> RLObservation:
        return super().observation_discrete_encode(np.asarray(val))

    def observation_continuous_encode(self, val: List[float]) -> RLObservation:
        return super().observation_continuous_encode(np.asarray(val))
