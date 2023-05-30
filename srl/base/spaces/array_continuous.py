import logging
import time
from typing import Any, List, Tuple, Union

import numpy as np

from srl.base.define import InvalidActionsType, RLTypes

from .box import SpaceBase

logger = logging.getLogger(__name__)


class ArrayContinuousSpace(SpaceBase[List[float]]):
    def __init__(
        self,
        size: int,
        low: Union[float, List[float], Tuple[float, ...], np.ndarray] = -np.inf,
        high: Union[float, List[float], Tuple[float, ...], np.ndarray] = np.inf,
    ) -> None:
        self._size = size
        self._low: np.ndarray = np.full((size,), low, dtype=np.float32) if np.isscalar(low) else np.asarray(low)
        self._high: np.ndarray = np.full((size,), high, dtype=np.float32) if np.isscalar(high) else np.asarray(high)

        assert len(self._low) == size
        assert len(self._high) == size
        assert self.low.shape == self.high.shape
        assert np.less_equal(self.low, self.high).all()

        self._is_inf = np.isinf(low) or np.isinf(high)
        self.division_tbl = None

    def sample(self, invalid_actions: InvalidActionsType = []) -> List[float]:
        if self._is_inf:
            # infの場合は正規分布に従う乱数
            return np.random.normal(size=(self._size,)).tolist()
        r = np.random.random_sample((self._size,))
        return (self._low + r * (self._high - self._low)).tolist()

    def convert(self, val: Any) -> List[float]:
        if isinstance(val, list):
            return [float(v) for v in val]
        elif isinstance(val, tuple):
            return [float(v) for v in val]
        elif isinstance(val, np.ndarray):
            return val.tolist()
        return [float(val) for _ in range(self._size)]

    def check_val(self, val: Any) -> bool:
        if not isinstance(val, list):
            return False
        if len(val) != self._size:
            return False
        for i in range(self._size):
            if not isinstance(val[i], float):
                return False
            if val[i] < self.low[i]:
                return False
            if val[i] > self.high[i]:
                return False
        return True

    @property
    def rl_type(self) -> RLTypes:
        return RLTypes.CONTINUOUS

    def get_default(self) -> List[float]:
        return [0.0 for _ in range(self._size)]

    def __eq__(self, o: "ArrayContinuousSpace") -> bool:
        return self._size == o._size and (self._low == o._low).all() and (self._high == o._high).all()

    def __str__(self) -> str:
        if self.division_tbl is None:
            s = ""
        else:
            s = f", division({self.n})"
        return f"ArrayContinuous({self._size}, range[{np.min(self.low)}, {np.max(self.high)}]){s}"

    # --- test
    def assert_params(self, true_size: int, true_low: np.ndarray, true_high: np.ndarray):
        assert self._size == true_size
        assert (self._low == true_low).all()
        assert (self._high == true_high).all()

    # --------------------------------------
    # create_division_tbl
    # --------------------------------------
    def create_division_tbl(self, division_num: int) -> None:
        if self._is_inf:  # infは定義できない
            return
        if division_num <= 0:
            return

        import itertools

        t0 = time.time()
        act_list = []
        for i in range(self._size):
            low = self._low[i]
            high = self._high[i]
            diff = (high - low) / (division_num - 1)
            act_list.append([float(low + diff * j) for j in range(division_num)])

        act_list = list(itertools.product(*act_list))
        self.division_tbl = np.array(act_list)
        n = len(self.division_tbl)

        logger.info(f"created division: {division_num}(n={n})({time.time()-t0:.3f}s)")

    # --------------------------------------
    # discrete
    # --------------------------------------
    @property
    def n(self) -> int:
        assert self.division_tbl is not None, "Call 'create_division_tbl(division_num)' first"
        return len(self.division_tbl)

    def encode_to_int(self, val: List[float]) -> int:
        assert self.division_tbl is not None, "Call 'create_division_tbl(division_num)' first"
        d = np.sum(np.abs(self.division_tbl - val), axis=1)
        return int(np.argmin(d))

    def decode_from_int(self, val: int) -> List[float]:
        if self.division_tbl is None:
            return [float(val) for _ in range(self._size)]
        else:
            return self.division_tbl[val].tolist()

    # --------------------------------------
    # discrete numpy
    # --------------------------------------
    def encode_to_int_np(self, val: List[float]) -> np.ndarray:
        if self.division_tbl is None:
            return np.round(val)
        else:
            # 分割してある場合
            n = self.encode_to_int(val)
            return np.array([n])

    def decode_from_int_np(self, val: np.ndarray) -> List[float]:
        if self.division_tbl is None:
            return val.astype(np.float32).tolist()
        else:
            return self.division_tbl[int(val[0])].tolist()

    # --------------------------------------
    # continuous list
    # --------------------------------------
    @property
    def list_size(self) -> int:
        return self._size

    @property
    def list_low(self) -> List[float]:
        return self._low.tolist()

    @property
    def list_high(self) -> List[float]:
        return self._high.tolist()

    def encode_to_list_float(self, val: List[float]) -> List[float]:
        return val

    def decode_from_list_float(self, val: List[float]) -> List[float]:
        return val

    # --------------------------------------
    # continuous numpy
    # --------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return (self._size,)

    @property
    def low(self) -> np.ndarray:
        return self._low

    @property
    def high(self) -> np.ndarray:
        return self._high

    def encode_to_np(self, val: List[float]) -> np.ndarray:
        return np.array(val, dtype=np.float32)

    def decode_from_np(self, val: np.ndarray) -> List[float]:
        return val.tolist()
