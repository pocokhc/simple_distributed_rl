import logging
import random
from typing import Any, List, Tuple, Union, cast

import numpy as np

from srl.base.define import InvalidActionsType, RLTypes

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

    def sample(self, invalid_actions: InvalidActionsType = []) -> List[int]:
        self._create_tbl()
        assert self.decode_tbl is not None

        valid_actions = []
        for a in self.decode_tbl:  # decode_tbl is all action
            if a not in invalid_actions:
                valid_actions.append(a)

        return list(random.choice(valid_actions))

    def convert(self, val: Any) -> List[int]:
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

    @property
    def rl_type(self) -> RLTypes:
        return RLTypes.DISCRETE

    def get_default(self) -> List[int]:
        return [0 for _ in range(self._size)]

    def __eq__(self, o: "ArrayDiscreteSpace") -> bool:
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

    # --- test
    def assert_params(self, true_size: int, true_low: List[int], true_high: List[int]):
        assert self._size == true_size
        assert self._low == true_low
        assert self._high == true_high

    # --------------------------------------
    # create_division_tbl
    # --------------------------------------
    def _create_tbl(self) -> None:
        if self.decode_tbl is not None:
            return
        import itertools

        if self._size > 10:
            logger.warning("It may take some time.")

        arr_list = [[a for a in range(self.low[i], self.high[i] + 1)] for i in range(self._size)]

        self.decode_tbl = list(itertools.product(*arr_list))
        self.encode_tbl = {}
        for i, v in enumerate(self.decode_tbl):
            self.encode_tbl[v] = i

    # --------------------------------------
    # discrete
    # --------------------------------------
    @property
    def n(self) -> int:
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
    # discrete numpy
    # --------------------------------------
    def encode_to_int_np(self, val: List[int]) -> np.ndarray:
        return np.array(val)

    def decode_from_int_np(self, val: np.ndarray) -> List[int]:
        return np.round(val).tolist()

    # --------------------------------------
    # continuous list
    # --------------------------------------
    @property
    def list_size(self) -> int:
        return self._size

    @property
    def list_low(self) -> List[float]:
        return cast(List[float], self._low)

    @property
    def list_high(self) -> List[float]:
        return cast(List[float], self._high)

    def encode_to_list_float(self, val: List[int]) -> List[float]:
        return [float(v) for v in val]

    def decode_from_list_float(self, val: List[float]) -> List[int]:
        return [int(round(v)) for v in val]

    # --------------------------------------
    # continuous numpy
    # --------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return (self._size,)

    @property
    def low(self) -> np.ndarray:
        return np.array(self._low)

    @property
    def high(self) -> np.ndarray:
        return np.array(self._high)

    def encode_to_np(self, val: List[int]) -> np.ndarray:
        return np.array(val, dtype=np.float32)

    def decode_from_np(self, val: np.ndarray) -> List[int]:
        return np.round(val).astype(np.int32).tolist()
