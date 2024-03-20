import logging
import random
import time
from typing import Any, List, Tuple, Union, cast

import numpy as np

from srl.base.define import SpaceTypes

from .space import SpaceBase

logger = logging.getLogger(__name__)


class ArrayDiscreteSpace(SpaceBase[List[int]]):
    def __init__(
        self,
        size: int,
        low: Union[int, List[int]],
        high: Union[int, List[int]],
    ) -> None:
        self._size = size
        assert isinstance(size, int)

        self._low = [low for _ in range(self._size)] if isinstance(low, int) else low
        assert len(self._low) == size
        self._low = [int(low) for low in self._low]

        self._high = [high for _ in range(self._size)] if isinstance(high, int) else high
        assert len(self._high) == size
        self._high = [int(h) for h in self._high]

        self.decode_tbl = None
        self.encode_tbl = None

    @property
    def size(self) -> int:
        return self._size

    @property
    def low(self) -> List[int]:
        return self._low

    @property
    def high(self) -> List[int]:
        return self._high

    @property
    def stype(self) -> SpaceTypes:
        return SpaceTypes.DISCRETE

    @property
    def dtype(self):
        return np.int64

    def sample(self, mask: List[List[int]] = []) -> List[int]:
        if len(mask) > 0:
            self._create_tbl()
            valid_acts = []
            for a in [k for k in self.encode_tbl.keys()]:
                f = True
                for m in mask:
                    if a == tuple(m):
                        f = False
                        break
                if f:
                    valid_acts.append(list(a))
            return random.choice(valid_acts)
        else:
            return [random.randint(self._low[i], self._high[i]) for i in range(self._size)]

    def get_valid_actions(self, mask: List[List[int]] = []) -> List[List[int]]:
        self._create_tbl()
        valid_acts = []
        for a in [k for k in self.encode_tbl.keys()]:
            f = True
            for m in mask:
                if a == tuple(m):
                    f = False
                    break
            if f:
                valid_acts.append(list(a))
        return valid_acts

    def sanitize(self, val: Any) -> List[int]:
        if isinstance(val, list):
            val = [int(np.round(v)) for v in val]
        elif isinstance(val, tuple):
            val = [int(np.round(v)) for v in val]
        elif isinstance(val, np.ndarray):
            val = val.round().astype(int).tolist()
        else:
            val = [int(np.round(val)) for _ in range(self._size)]
        for i in range(self._size):
            if val[i] < self._low[i]:
                val[i] = self._low[i]
            elif val[i] > self._high[i]:
                val[i] = self._high[i]
        return val

    def check_val(self, val: Any) -> bool:
        if not isinstance(val, list):
            return False
        if len(val) != self._size:
            return False
        for i in range(self._size):
            if not isinstance(val[i], int):
                return False
            if val[i] < self.low[i]:
                return False
            if val[i] > self.high[i]:
                return False
        return True

    def to_str(self, val: List[int]) -> str:
        return ",".join([str(v) for v in val])

    def get_default(self) -> List[int]:
        return [0 if self._low[i] <= 0 <= self._high[i] else self._low[i] for i in range(self._size)]

    def copy(self) -> "ArrayDiscreteSpace":
        o = ArrayDiscreteSpace(self._size, self._low, self._high)
        o.decode_tbl = self.decode_tbl
        o.encode_tbl = self.encode_tbl
        return o

    def __eq__(self, o: "ArrayDiscreteSpace") -> bool:
        if not isinstance(o, ArrayDiscreteSpace):
            return False
        if self._size != o._size:
            return False
        if self.low is None:
            if o.low is not None:
                return False
        else:
            if o.low is None:
                return False
            for i in range(self._size):
                if self.low[i] != o.low[i]:
                    return False
        if self.high is None:
            if o.high is not None:
                return False
        else:
            if o.high is None:
                return False
            for i in range(self._size):
                if self.high[i] != o.high[i]:
                    return False
        return True

    def __str__(self) -> str:
        return f"ArrayDiscrete({self._size}, range[{int(np.min(self.low))}, {int(np.max(self.high))}])"

    # --- stack
    def create_stack_space(self, length: int):
        return ArrayDiscreteSpace(
            length * self._size,
            length * self._low,
            length * self._high,
        )

    def encode_stack(self, val: List[List[int]]) -> List[int]:
        return [e for sublist in val for e in sublist]

    # --------------------------------------
    # create_tbl
    # --------------------------------------
    def _create_tbl(self) -> None:
        if self.decode_tbl is not None:
            return
        import itertools

        t0 = time.time()
        arr_list = [[a for a in range(self.low[i], self.high[i] + 1)] for i in range(self._size)]
        self.decode_tbl = list(itertools.product(*arr_list))
        self.encode_tbl = {}
        for i, v in enumerate(self.decode_tbl):
            self.encode_tbl[v] = i
        logger.info(f"create table time: {time.time() - t0:.1f}s")

    # --------------------------------------
    # action discrete
    # --------------------------------------
    @property
    def int_size(self) -> int:
        self._create_tbl()
        assert self.decode_tbl is not None
        return len(self.decode_tbl)

    def encode_to_int(self, val: List[int]) -> int:
        self._create_tbl()
        return self.encode_tbl[tuple(val)]

    def decode_from_int(self, val: int) -> List[int]:
        self._create_tbl()
        assert self.decode_tbl is not None
        return list(self.decode_tbl[val])

    # --------------------------------------
    # observation discrete
    # --------------------------------------
    @property
    def list_int_size(self) -> int:
        return self._size

    @property
    def list_int_low(self) -> List[int]:
        return self._low

    @property
    def list_int_high(self) -> List[int]:
        return self._high

    def encode_to_list_int(self, val: List[int]) -> List[int]:
        return val

    def decode_from_list_int(self, val: List[int]) -> List[int]:
        return val

    # --------------------------------------
    # action continuous
    # --------------------------------------
    @property
    def list_float_size(self) -> int:
        return self._size

    @property
    def list_float_low(self) -> List[float]:
        return cast(List[float], self._low)

    @property
    def list_float_high(self) -> List[float]:
        return cast(List[float], self._high)

    def encode_to_list_float(self, val: List[int]) -> List[float]:
        return [float(v) for v in val]

    def decode_from_list_float(self, val: List[float]) -> List[int]:
        return [int(round(v)) for v in val]

    # --------------------------------------
    # observation continuous, image
    # --------------------------------------
    @property
    def np_shape(self) -> Tuple[int, ...]:
        return (self._size,)

    @property
    def np_low(self) -> np.ndarray:
        return np.array(self._low)

    @property
    def np_high(self) -> np.ndarray:
        return np.array(self._high)

    def encode_to_np(self, val: List[int], dtype) -> np.ndarray:
        return np.array(val, dtype=dtype)

    def decode_from_np(self, val: np.ndarray) -> List[int]:
        return np.round(val).astype(np.int64).tolist()
