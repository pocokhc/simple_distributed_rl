import logging
import random
from typing import Any, List, Tuple

import numpy as np

from srl.base.define import EnvTypes

from .space import SpaceBase

logger = logging.getLogger(__name__)


class DiscreteSpace(SpaceBase[int]):
    def __init__(self, n: int, start: int = 0) -> None:
        assert n > 0
        self._n = n
        self._start = start

        self._log_sanitize_count_low = 0
        self._log_sanitize_count_high = 0

    @property
    def base_env_type(self) -> EnvTypes:
        return EnvTypes.DISCRETE

    @property
    def dtype(self):
        return np.uint64 if self._start >= 0 else np.int64

    def sample(self, mask: List[int] = []) -> int:
        assert len(mask) < self._n, f"No valid actions. {mask}"
        acts = [a + self._start for a in range(self.n)]
        return random.choice([a for a in acts if a not in mask])

    def get_valid_actions(self, mask: List[int] = []) -> List[int]:
        acts = [a + self._start for a in range(self.n)]
        return [a for a in acts if a not in mask]

    def sanitize(self, val: Any) -> int:
        if isinstance(val, list):
            val = round(val[0])
        elif isinstance(val, tuple):
            val = round(val[0])
        else:
            val = round(val)
        if val < self._start:
            if self._log_sanitize_count_low < 5:
                logger.info(f"The value was changed with sanitize. {val} -> {self._start}")
                self._log_sanitize_count_low += 1
            val = self._start
        elif val >= self.n + self._start:
            _old_val = val
            val = self.n - 1 + self._start
            if self._log_sanitize_count_high < 5:
                logger.info(f"The value was changed with sanitize. {_old_val} -> {val}")
                self._log_sanitize_count_high += 1
        return val

    def check_val(self, val: Any) -> bool:
        if not isinstance(val, int):
            return False
        if val < self._start:
            return False
        if val >= self.n + self._start:
            return False
        return True

    def get_default(self) -> int:
        return self._start

    def copy(self) -> "DiscreteSpace":
        return DiscreteSpace(self._n, self._start)

    def __eq__(self, o: "DiscreteSpace") -> bool:
        return self._n == o._n and self._start == o._start

    def __str__(self) -> str:
        return f"Discrete({self._n}, start={self._start})"

    # --------------------------------------
    # action discrete
    # --------------------------------------
    @property
    def n(self) -> int:
        return self._n

    def encode_to_int(self, val: int) -> int:
        return val - self._start

    def decode_from_int(self, val: int) -> int:
        return val + self._start

    # --------------------------------------
    # observation discrete
    # --------------------------------------
    def encode_to_list_int(self, val: int) -> List[int]:
        return [val - self._start]

    def decode_from_list_int(self, val: List[int]) -> int:
        return int(round(val[0])) + self._start

    # --------------------------------------
    # action continuous
    # --------------------------------------
    @property
    def list_size(self) -> int:
        return 1

    @property
    def list_low(self) -> List[float]:
        return [0]

    @property
    def list_high(self) -> List[float]:
        return [self.n - 1]

    def encode_to_list_float(self, val: int) -> List[float]:
        return [float(val - self._start)]

    def decode_from_list_float(self, val: List[float]) -> int:
        return int(round(val[0])) + self._start

    # --------------------------------------
    # observation continuous, image
    # --------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return (1,)

    @property
    def low(self) -> np.ndarray:
        return np.array([0])

    @property
    def high(self) -> np.ndarray:
        return np.array([self.n - 1])

    def encode_to_np(self, val: int, dtype) -> np.ndarray:
        return np.array([val - self._start], dtype=dtype)

    def decode_from_np(self, val: np.ndarray) -> int:
        return int(round(val[0])) + self._start
