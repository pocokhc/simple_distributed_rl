import itertools
import logging
from typing import Any, List, Tuple, Union

import numpy as np
from srl.base.define import ContinuousAction, DiscreteAction, DiscreteSpaceType, RLObservation
from srl.base.env.base import SpaceBase

logger = logging.getLogger(__name__)


class BoxSpace(SpaceBase[np.ndarray]):
    def __init__(
        self,
        shape: Tuple[int, ...],
        low: Union[float, np.ndarray] = -np.inf,
        high: Union[float, np.ndarray] = np.inf,
    ) -> None:
        self._low: np.ndarray = np.full(shape, low, dtype=np.float32) if np.isscalar(low) else low
        self._high: np.ndarray = np.full(shape, high, dtype=np.float32) if np.isscalar(high) else high
        self._shape = shape

        assert self.shape == self.high.shape
        assert self.low.shape == self.high.shape
        assert np.less(self.low, self.high).all()

        self._is_inf = np.isinf(low).any() or np.isinf(high).any()
        self._is_division = False
        self._n = 0

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

    def sample(self, invalid_actions: List[DiscreteSpaceType] = []) -> np.ndarray:
        if self._is_inf:
            # infの場合は正規分布に従う乱数
            return np.random.normal(size=self.shape)
        r = np.random.random_sample(self.shape)
        return self.low + r * (self.high - self.low)

    def check_val(self, val: Any) -> bool:
        if not isinstance(val, np.ndarray):
            return False
        if self.shape != val.shape:
            return False
        if (val < self.low).any():
            return False
        if (val > self.high).any():
            return False
        return True

    def __eq__(self, o: object) -> bool:
        return self.shape == o.shape and (self.low == o.low).all() and (self.high == o.high).all()

    def __str__(self) -> str:
        return f"Box({self.shape}, {np.min(self.low)}, {np.max(self.high)})"

    # --- discrete
    def set_action_division(self, division_num: int) -> None:
        if self._is_inf:
            return

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
        self.action_tbl = np.reshape(act_list, (-1,) + self.shape)
        self._n = len(self.action_tbl)

    # --- action discrete
    def get_action_discrete_info(self) -> int:
        return self._n

    def action_discrete_encode(self, val: np.ndarray) -> DiscreteAction:
        if self._is_inf:  # infは定義できない
            raise NotImplementedError
        # ユークリッド距離で一番近いアクションを選択
        d = (self.action_tbl - val).reshape((self.action_tbl.shape[0], -1))
        d = np.linalg.norm(d, axis=1)
        return np.argmin(d)

    def action_discrete_decode(self, val: DiscreteAction) -> np.ndarray:
        if self._is_inf:  # infは定義できない
            return np.full(self.shape, val, dtype=np.float32)
        return self.action_tbl[val]

    # --- action continuous
    def get_action_continuous_info(self) -> Tuple[int, np.ndarray, np.ndarray]:
        low = self.low.flatten()
        high = self.high.flatten()
        return len(low), low, high

    # def action_continuous_encode(self, val: np.ndarray) -> ContinuousAction:
    #    return val.flatten().tolist()

    def action_continuous_decode(self, val: ContinuousAction) -> np.ndarray:
        return np.asarray(val).reshape(self.shape)

    # --- observation
    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self.shape

    def observation_discrete_encode(self, val: np.ndarray) -> RLObservation:
        return np.round(val).astype(int)

    def observation_continuous_encode(self, val: np.ndarray) -> RLObservation:
        return val.astype(float)
