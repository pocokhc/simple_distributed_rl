from typing import Any, List, Tuple

import numpy as np

from srl.base.define import InvalidActionsType, RLTypes

from .space import SpaceBase


class DiscreteSpace(SpaceBase[int]):
    def __init__(self, n: int) -> None:
        self._n = n

        assert n > 0

    def sample(self, invalid_actions: InvalidActionsType = []) -> int:
        assert len(invalid_actions) < self.n, f"No valid actions. {invalid_actions}"
        return int(np.random.choice([a for a in range(self.n) if a not in invalid_actions]))

    def convert(self, val: Any) -> int:
        if isinstance(val, list):
            val = round(val[0])
        elif isinstance(val, tuple):
            val = round(val[0])
        else:
            val = round(val)
        if val < 0:
            val = 0
        elif val >= self.n:
            val = self.n - 1
        return val

    def check_val(self, val: Any) -> bool:
        if not isinstance(val, int):
            return False
        if val < 0:
            return False
        if val >= self.n:
            return False
        return True

    @property
    def rl_type(self) -> RLTypes:
        return RLTypes.DISCRETE

    def get_default(self) -> int:
        return 0

    def __eq__(self, o: "DiscreteSpace") -> bool:
        return self.n == o.n

    def __str__(self) -> str:
        return f"Discrete({self.n})"

    # --------------------------------------
    # int
    # --------------------------------------
    @property
    def n(self) -> int:
        return self._n

    def encode_to_int(self, val: int) -> int:
        return val

    def decode_from_int(self, val: int) -> int:
        return val

    # --------------------------------------
    # discrete numpy
    # --------------------------------------
    def encode_to_int_np(self, val: int) -> np.ndarray:
        return np.array([val])

    def decode_from_int_np(self, val: np.ndarray) -> int:
        return int(round(val[0]))

    # --------------------------------------
    # continuous list
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
        return [float(val)]

    def decode_from_list_float(self, val: List[float]) -> int:
        return int(round(val[0]))

    # --------------------------------------
    # continuous numpy
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

    def encode_to_np(self, val: int) -> np.ndarray:
        return np.array([val], dtype=np.float32)

    def decode_from_np(self, val: np.ndarray) -> int:
        return int(round(val[0]))
