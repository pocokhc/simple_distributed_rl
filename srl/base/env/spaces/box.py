import itertools
import logging
from typing import List, SupportsFloat, Tuple, Union

import numpy as np
from srl.base.define import InvalidAction
from srl.base.env.base import SpaceBase

logger = logging.getLogger(__name__)


class BoxSpace(SpaceBase[np.ndarray]):
    def __init__(
        self,
        low: Union[SupportsFloat, np.ndarray],
        high: Union[SupportsFloat, np.ndarray],
        shape: Tuple[int, ...],
    ) -> None:
        self._low = np.full(shape, low, dtype=float) if np.isscalar(low) else low
        self._high = np.full(shape, high, dtype=float) if np.isscalar(high) else high
        self._shape = shape
        self._array_len = len(self._low.flatten())

        assert isinstance(self._shape, tuple), f"shape is a tuple format. type=({type(shape)})"

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def low(self) -> np.ndarray:
        return self._low

    @property
    def high(self) -> np.ndarray:
        return self._high

    def sample(self, invalid_actions: List[InvalidAction] = []) -> np.ndarray:
        r = np.random.random_sample(self.shape)
        return self.low + r * (self.high - self.low)

    # --- action discrete
    def set_division(self, division_num: int) -> None:
        low_flatten = self.low.flatten()
        high_flatten = self.high.flatten()

        if len(low_flatten) ** division_num > 100_000:
            logger.warn("It may take some time.")

        act_list = []
        for i in range(len(low_flatten)):
            act = []
            for j in range(division_num):
                low = low_flatten[i]
                high = high_flatten[i]
                diff = (high - low) / (division_num - 1)

                a = low + diff * j
                act.append(a)
            act_list.append(act)

        act_list = list(itertools.product(*act_list))
        self.action_tbl = np.reshape(act_list, (-1,) + self.shape).tolist()
        self._n = len(self.action_tbl)

    def get_action_discrete_info(self) -> int:
        return self._n

    def action_discrete_encode(self, val: np.ndarray) -> int:
        raise NotImplementedError

    def action_discrete_decode(self, val: int) -> np.ndarray:
        return self.action_tbl[val]

    # --- action continuous
    def get_action_continuous_info(self) -> Tuple[int, np.ndarray, np.ndarray]:
        return self._array_len, self.low, self.high

    def action_continuous_encode(self, val: np.ndarray) -> List[float]:
        return val.flatten().tolist()

    def action_continuous_decode(self, val: List[float]) -> np.ndarray:
        return np.asarray(val).reshape(self.shape)

    # --- observation discrete
    def get_observation_discrete_info(self) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray]:
        return self.shape, self.low, self.high

    def observation_discrete_encode(self, val: np.ndarray) -> np.ndarray:
        return np.round(val)

    # --- observation continuous
    def get_observation_continuous_info(self) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray]:
        return self.shape, self.low, self.high

    def observation_continuous_encode(self, val: np.ndarray) -> np.ndarray:
        return val
