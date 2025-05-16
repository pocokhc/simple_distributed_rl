import logging
import random
from typing import Any, List, Tuple

import numpy as np

from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.space import SpaceBase

logger = logging.getLogger(__name__)


class ContinuousSpace(SpaceBase[float]):
    def __init__(self, low: float = -np.inf, high: float = np.inf, dtype=np.float32) -> None:
        self._low = float(low)
        self._high = float(high)
        self._dtype = dtype

        assert self.low <= self.high

        self._is_inf = np.isinf(low) or np.isinf(high)
        self.division_tbl = None

        self._log_sanitize_count_low = 0
        self._log_sanitize_count_high = 0

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def stype(self) -> SpaceTypes:
        return SpaceTypes.CONTINUOUS

    @property
    def dtype(self):
        return self._dtype

    def sample(self, mask: List[float] = []) -> float:
        if len(mask) > 0:
            logger.info(f"mask is not support: {mask}")
        if self._is_inf:
            # infの場合は正規分布に従う乱数
            return float(np.random.normal(size=1))
        r = random.random()
        return self._low + r * (self._high - self._low)

    def sanitize(self, val: Any) -> float:
        if isinstance(val, list):
            val = float(val[0])
        elif isinstance(val, tuple):
            val = float(val[0])
        else:
            val = float(val)
        if val < self._low:
            if self._log_sanitize_count_low < 5:
                logger.info(f"The value was changed with sanitize. {val} -> {self._low}")
                self._log_sanitize_count_low += 1
            val = self._low
        elif val > self._high:
            if self._log_sanitize_count_high < 5:
                logger.info(f"The value was changed with sanitize. {val} -> {self._high}")
                self._log_sanitize_count_high += 1
            val = self._high
        return val

    def check_val(self, val: Any) -> bool:
        if not isinstance(val, float):
            return False
        if val < self._low:
            return False
        if val > self._high:
            return False
        return True

    def to_str(self, val: float) -> str:
        return str(int(val) if val.is_integer() else val)

    def get_default(self) -> float:
        return 0.0 if self._low <= 0 <= self._high else self._low

    def copy(self, **kwargs) -> "ContinuousSpace":
        keys = ["low", "high", "dtype"]
        args = [kwargs.get(key, getattr(self, f"_{key}")) for key in keys]
        o = ContinuousSpace(*args)
        o.division_tbl = self.division_tbl
        return o

    def copy_value(self, v: float) -> float:
        return v

    def __eq__(self, o: "ContinuousSpace") -> bool:
        if not isinstance(o, ContinuousSpace):
            return False
        return (self._low == o._low) and (self._high == o._high)

    def __str__(self) -> str:
        if self.division_tbl is None:
            s = ")"
        else:
            s = f", division({self.int_size})"
        return f"Continuous({self.low} - {self.high}{s}"

    # --- stack
    def create_stack_space(self, length: int):
        from srl.base.spaces.array_continuous import ArrayContinuousSpace

        return ArrayContinuousSpace(length, self._low, self._high)

    def encode_stack(self, val: List[float]) -> List[float]:
        return val

    # --------------------------------------
    # create_division_tbl
    # --------------------------------------
    def create_division_tbl(
        self,
        division_num: int,
        max_size: int = 100_000,
        max_byte: int = 1024 * 1024 * 1024,
    ) -> None:
        if self.division_tbl is not None:
            return
        if self._is_inf:  # infは定義できない
            return
        if division_num <= 0:
            return

        diff = (self._high - self._low) / (division_num - 1)
        self.division_tbl = np.array([self._low + diff * j for j in range(division_num)])
        n = len(self.division_tbl)

        logger.info(f"created division: {division_num}(n={n})")

    # --------------------------------------
    # action discrete
    # --------------------------------------
    @property
    def int_size(self) -> int:
        assert self.division_tbl is not None, "Call 'create_division_tbl(division_num)' first"
        return len(self.division_tbl)

    def encode_to_int(self, val: float) -> int:
        if self.division_tbl is None:
            return int(round(val))
        else:
            # 一番近いもの
            d = np.abs(self.division_tbl - val)
            return int(np.argmin(d))

    def decode_from_int(self, val: int) -> float:
        if self.division_tbl is None:
            return float(val)
        else:
            return self.division_tbl[val]

    # --------------------------------------
    # observation discrete
    # --------------------------------------
    @property
    def list_int_size(self) -> int:
        return 1

    @property
    def list_int_low(self) -> List[int]:
        if self.division_tbl is None:
            return [int(round(self._low))]
        else:
            return [0]

    @property
    def list_int_high(self) -> List[int]:
        if self.division_tbl is None:
            return [int(round(self._high))]
        else:
            return [len(self.division_tbl)]

    def encode_to_list_int(self, val: float) -> List[int]:
        if self.division_tbl is None:
            return [int(round(val))]
        else:
            return [self.encode_to_int(val)]

    def decode_from_list_int(self, val: List[int]) -> float:
        if self.division_tbl is None:
            return float(val[0])
        else:
            return self.division_tbl[val[0]]

    # --------------------------------------
    # action continuous
    # --------------------------------------
    @property
    def list_float_size(self) -> int:
        return 1

    @property
    def list_float_low(self) -> List[float]:
        return [self._low]

    @property
    def list_float_high(self) -> List[float]:
        return [self._high]

    def encode_to_list_float(self, val: float) -> List[float]:
        return [val]

    def decode_from_list_float(self, val: List[float]) -> float:
        return val[0]

    # --------------------------------------
    # observation continuous, image
    # --------------------------------------
    @property
    def np_shape(self) -> Tuple[int, ...]:
        return (1,)

    @property
    def np_low(self) -> np.ndarray:
        return np.array([self._low])

    @property
    def np_high(self) -> np.ndarray:
        return np.array([self._high])

    def encode_to_np(self, val: float, dtype) -> np.ndarray:
        return np.array([val], dtype=dtype)

    def decode_from_np(self, val: np.ndarray) -> float:
        return float(val[0])

    # --------------------------------------
    # spaces
    # --------------------------------------
    def get_encode_type_list(self):
        priority_list = [
            RLBaseTypes.CONTINUOUS,
            RLBaseTypes.ARRAY_CONTINUOUS,
            RLBaseTypes.BOX,
            RLBaseTypes.ARRAY_DISCRETE,
        ]
        exclude_list = []
        return priority_list, exclude_list

    # --- DiscreteSpace
    def create_encode_space_DiscreteSpace(self):
        from srl.base.spaces.discrete import DiscreteSpace

        return DiscreteSpace(self.int_size)  # startは0

    def encode_to_space_DiscreteSpace(self, val: float) -> int:
        if self.division_tbl is None:
            return int(round(val))
        else:
            # 一番近いもの
            d = np.abs(self.division_tbl - val)
            return int(np.argmin(d))

    def decode_from_space_DiscreteSpace(self, val: int) -> float:
        if self.division_tbl is None:
            return float(val)
        else:
            return float(self.division_tbl[val])

    # --- ArrayDiscreteSpace
    def create_encode_space_ArrayDiscreteSpace(self):
        from srl.base.spaces.array_discrete import ArrayDiscreteSpace

        return ArrayDiscreteSpace(self.list_int_size, self.list_int_low, self.list_int_high)

    def encode_to_space_ArrayDiscreteSpace(self, val: float) -> List[int]:
        if self.division_tbl is None:
            return [int(round(val))]
        else:
            return [self.encode_to_int(val)]

    def decode_from_space_ArrayDiscreteSpace(self, val: List[int]) -> float:
        if self.division_tbl is None:
            return float(val[0])
        else:
            return float(self.division_tbl[val[0]])

    # --- ContinuousSpace
    def create_encode_space_ContinuousSpace(self):
        return self.copy()

    def encode_to_space_ContinuousSpace(self, val: float) -> float:
        return val

    def decode_from_space_ContinuousSpace(self, val: float) -> float:
        return float(val)

    # --- ArrayContinuousSpace
    def create_encode_space_ArrayContinuousSpace(self):
        from srl.base.spaces.array_continuous import ArrayContinuousSpace

        return ArrayContinuousSpace(self.list_float_size, self.list_float_low, self.list_float_high)

    def encode_to_space_ArrayContinuousSpace(self, val: float) -> List[float]:
        return [val]

    def decode_from_space_ArrayContinuousSpace(self, val: List[float]) -> float:
        return float(val[0])

    # --- Box
    def create_encode_space_Box(self, space_type: RLBaseTypes, np_dtype):
        from srl.base.spaces.box import BoxSpace

        # TODO: Box Image

        return BoxSpace(self.np_shape, self.np_low, self.np_high, np_dtype, stype=SpaceTypes.CONTINUOUS)

    def encode_to_space_Box(self, val: float, space) -> np.ndarray:
        return np.array([val], space.dtype)

    def decode_from_space_Box(self, val: np.ndarray, space) -> float:
        return float(val[0])

    # --- TextSpace
    def create_encode_space_TextSpace(self):
        raise NotSupportedError()

    def encode_to_space_TextSpace(self, val: float) -> str:
        return str(val)

    def decode_from_space_TextSpace(self, val: str) -> float:
        return float(val)
