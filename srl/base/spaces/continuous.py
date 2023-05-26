import logging
import random
from typing import Any, List, Tuple

import numpy as np

from srl.base.define import InvalidActionsType, RLTypes

from .box import SpaceBase

logger = logging.getLogger(__name__)


class ContinuousSpace(SpaceBase[float]):
    def __init__(
        self,
        low: float = -np.inf,
        high: float = np.inf,
    ) -> None:
        self._low = low
        self._high = high

        assert self.low.shape == self.high.shape
        assert self.low < self.high

        self._is_inf = np.isinf(low) or np.isinf(high)
        self.division_tbl = None

    def sample(self, invalid_actions: InvalidActionsType = []) -> float:
        if self._is_inf:
            # infの場合は正規分布に従う乱数
            return float(np.random.normal(size=1))
        r = random.random()
        return self._low + r * (self._high - self._low)

    def convert(self, val: Any) -> float:
        if isinstance(val, list):
            return float(val[0])
        elif isinstance(val, tuple):
            return float(val[0])
        return float(val)

    def check_val(self, val: Any) -> bool:
        if not isinstance(val, float):
            return False
        if val < self.low:
            return False
        if val > self.high:
            return False
        return True

    @property
    def rl_type(self) -> RLTypes:
        return RLTypes.CONTINUOUS

    def get_default(self) -> float:
        return 0.0

    def __eq__(self, o: "ContinuousSpace") -> bool:
        return (self._low == o._low) and (self._high == o._high)

    def __str__(self) -> str:
        return f"Continuous({self.low} - {self.high})"

    # --------------------------------------
    # create_division_tbl
    # --------------------------------------
    def create_division_tbl(self, division_num: int) -> None:
        if self._is_inf:  # infは定義できない
            return
        if division_num <= 0:
            return

        diff = (self._high - self._low) / (division_num - 1)
        self.division_tbl = np.array([self._low + diff * j for j in range(division_num)])
        n = len(self.division_tbl)

        logger.info(f"created division: {division_num}(n={n})")

    # --------------------------------------
    # discrete
    # --------------------------------------
    @property
    def n(self) -> int:
        assert self.division_tbl is not None, "Call 'create_division_tbl(division_num)' first"
        return len(self.division_tbl)

    def encode_to_int(self, val: float) -> int:
        assert self.division_tbl is not None, "Call 'create_division_tbl(division_num)' first"
        # 一番近いもの
        d = np.abs(self.division_tbl - val)
        return int(np.argmin(d))

    def decode_from_int(self, val: int) -> float:
        if self.division_tbl is None:
            return float(val)
        else:
            return self.division_tbl[val]

    # --------------------------------------
    # discrete numpy
    # --------------------------------------
    def encode_to_int_np(self, val: float) -> np.ndarray:
        if self.division_tbl is None:
            return np.round([val])
        else:
            n = self.encode_to_int(val)
            return np.array([n])

    def decode_from_int_np(self, val: np.ndarray) -> float:
        if self.division_tbl is None:
            return float(val[0])
        else:
            return self.division_tbl[int(val[0])]

    # --------------------------------------
    # continuous list
    # --------------------------------------
    @property
    def list_size(self) -> int:
        return 1

    @property
    def list_low(self) -> List[float]:
        return [self._low]

    @property
    def list_high(self) -> List[float]:
        return [self._high]

    def encode_to_list_float(self, val: float) -> List[float]:
        return [val]

    def decode_from_list_float(self, val: List[float]) -> float:
        return val[0]

    # --------------------------------------
    # continuous numpy
    # --------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return (1,)

    @property
    def low(self) -> np.ndarray:
        return np.array([self._low])

    @property
    def high(self) -> np.ndarray:
        return np.array([self._high])

    def encode_to_np(self, val: float) -> np.ndarray:
        return np.array([val], dtype=np.float32)

    def decode_from_np(self, val: np.ndarray) -> float:
        return float(val[0])
