import logging
import time
from typing import Any, List, Tuple, Union

import numpy as np

from srl.base.define import InvalidActionsType, RLTypes

from .space import SpaceBase

logger = logging.getLogger(__name__)


class BoxSpace(SpaceBase[np.ndarray]):
    def __init__(
        self,
        shape: Union[List[int], Tuple[int, ...]],
        low: Union[float, List[float], Tuple[float, ...], np.ndarray] = -np.inf,
        high: Union[float, List[float], Tuple[float, ...], np.ndarray] = np.inf,
    ) -> None:
        self._low: np.ndarray = np.full(shape, low, dtype=np.float32) if np.isscalar(low) else np.asarray(low)
        self._high: np.ndarray = np.full(shape, high, dtype=np.float32) if np.isscalar(high) else np.asarray(high)
        self._shape = tuple(shape)

        assert self.shape == self.high.shape
        assert self.low.shape == self.high.shape
        assert np.less_equal(self.low, self.high).all()

        self._is_inf = np.isinf(low).any() or np.isinf(high).any()
        self.division_tbl = None

    def sample(self, invalid_actions: InvalidActionsType = []) -> np.ndarray:
        if self._is_inf:
            # infの場合は正規分布に従う乱数
            return np.random.normal(size=self.shape)
        r = np.random.random_sample(self.shape)
        return self.low + r * (self.high - self.low)

    def convert(self, val: Any) -> np.ndarray:
        return np.clip(val, self._low, self._high).astype(np.float32)

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

    @property
    def rl_type(self) -> RLTypes:
        return RLTypes.CONTINUOUS

    def get_default(self) -> np.ndarray:
        return np.zeros(self.shape)

    def __eq__(self, o: "BoxSpace") -> bool:
        return self.shape == o.shape and (self.low == o.low).all() and (self.high == o.high).all()

    def __str__(self) -> str:
        if self.division_tbl is None:
            s = ""
        else:
            s = f", division({self.n})"
        return f"Box({self.shape}, range[{np.min(self.low)}, {np.max(self.high)}]){s}"

    # --- test
    def assert_params(self, true_shape: Tuple[int, ...], true_low: np.ndarray, true_high: np.ndarray):
        assert self.shape == true_shape
        assert (self.low == true_low).all()
        assert (self.high == true_high).all()

    # --------------------------------------
    # create_division_tbl
    # --------------------------------------
    def create_division_tbl(self, division_num: int) -> None:
        if self._is_inf:  # infは定義できない
            return
        if division_num <= 0:
            return

        import itertools

        low_flatten = self.low.flatten()
        high_flatten = self.high.flatten()

        if len(low_flatten) ** division_num > 100_000:
            logger.warn("It may take some time.")

        t0 = time.time()
        act_list = []
        for i in range(len(low_flatten)):
            low = low_flatten[i]
            high = high_flatten[i]
            diff = (high - low) / (division_num - 1)
            act_list.append([low + diff * j for j in range(division_num)])

        act_list = list(itertools.product(*act_list))
        self.division_tbl = np.reshape(act_list, (-1,) + self.shape)
        n = len(self.division_tbl)

        logger.info(f"created division: {division_num}(n={n})({time.time()-t0:.3f}s)")

    # --------------------------------------
    # discrete
    # --------------------------------------
    @property
    def n(self) -> int:
        assert self.division_tbl is not None, "Call 'create_division_tbl(division_num)' first"
        return len(self.division_tbl)

    def encode_to_int(self, val: np.ndarray) -> int:
        assert self.division_tbl is not None, "Call 'create_division_tbl(division_num)' first"
        # ユークリッド距離で一番近いものを選択
        d = (self.division_tbl - val).reshape((self.division_tbl.shape[0], -1))
        d = np.linalg.norm(d, axis=1)
        return int(np.argmin(d))

    def decode_from_int(self, val: int) -> np.ndarray:
        if self.division_tbl is None:
            return np.full(self.shape, val, dtype=np.float32)
        else:
            return self.division_tbl[val]

    # --------------------------------------
    # discrete numpy
    # --------------------------------------
    def encode_to_int_np(self, val: np.ndarray) -> np.ndarray:
        if self.division_tbl is None:
            # 分割してない場合は、roundで丸めるだけ
            return np.round(val)
        else:
            # 分割してある場合
            n = self.encode_to_int(val)
            return np.array([n])

    def decode_from_int_np(self, val: np.ndarray) -> np.ndarray:
        if self.division_tbl is None:
            return val
        else:
            return self.division_tbl[int(val[0])]

    # --------------------------------------
    # continuous list
    # --------------------------------------
    @property
    def list_size(self) -> int:
        return len(self.low.flatten())

    @property
    def list_low(self) -> List[float]:
        return self.low.flatten().tolist()

    @property
    def list_high(self) -> List[float]:
        return self.high.flatten().tolist()

    def encode_to_list_float(self, val: np.ndarray) -> List[float]:
        return val.flatten().tolist()

    def decode_from_list_float(self, val: List[float]) -> np.ndarray:
        return np.asarray(val).reshape(self.shape)

    # --------------------------------------
    # continuous numpy
    # --------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def low(self) -> np.ndarray:
        return self._low

    @property
    def high(self) -> np.ndarray:
        return self._high

    def encode_to_np(self, val: np.ndarray) -> np.ndarray:
        return val.astype(np.float32)

    def decode_from_np(self, val: np.ndarray) -> np.ndarray:
        return val
