import random
from typing import List, Tuple, Union

import numpy as np
from srl.base.define import ContinuousAction, DiscreteAction, DiscreteSpaceType, RLObservation
from srl.base.env.spaces.box import BoxSpace


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

    def __str__(self) -> str:
        return f"ArrayContinuous({self.size}, {self.low}, {self.high})"

    # --- action discrete
    def get_action_discrete_info(self) -> int:
        return self._n

    def action_discrete_encode(self, val: List[float]) -> DiscreteAction:
        raise NotImplementedError

    def action_discrete_decode(self, val: DiscreteAction) -> List[float]:
        return super().action_discrete_decode(val).tolist()

    # --- action continuous
    def get_action_continuous_info(self) -> Tuple[int, np.ndarray, np.ndarray]:
        return super().get_action_continuous_info()

    # def action_continuous_encode(self, val: float) -> ContinuousAction:
    #    raise NotImplementedError

    def action_continuous_decode(self, val: ContinuousAction) -> List[float]:
        return val

    # --- observation discrete
    def get_observation_discrete_info(self) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray]:
        return super().get_observation_discrete_info()

    def observation_discrete_encode(self, val: List[float]) -> RLObservation:
        return super().observation_discrete_encode(np.asarray(val))

    # --- observation continuous
    def get_observation_continuous_info(self) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray]:
        return super().get_observation_continuous_info()

    def observation_continuous_encode(self, val: List[float]) -> RLObservation:
        return super().observation_continuous_encode(np.asarray(val))
