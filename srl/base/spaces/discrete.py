import logging
import random
from typing import TYPE_CHECKING, Any, List

import numpy as np

from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.spaces.space import SpaceBase, SpaceEncodeOptions

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from srl.base.spaces.box import BoxSpace


class DiscreteSpace(SpaceBase[int]):
    def __init__(self, n: int, start: int = 0) -> None:
        super().__init__()
        assert n > 0
        self._n = n
        self._start = start

        self._log_sanitize_count_low = 0
        self._log_sanitize_count_high = 0

    @property
    def n(self):
        return self._n

    @property
    def start(self):
        return self._start

    @property
    def name(self) -> str:
        return "Discrete"

    @property
    def stype(self) -> SpaceTypes:
        return SpaceTypes.DISCRETE

    @property
    def dtype(self):
        return np.uint64 if self._start >= 0 else np.int64

    def sample(self, mask: List[int] = []) -> int:
        assert len(mask) < self._n, f"No valid actions. {mask}"
        acts = [a + self._start for a in range(self._n)]
        return random.choice([a for a in acts if a not in mask])

    def get_valid_actions(self, masks: List[int] = []) -> List[int]:
        acts = [a + self._start for a in range(self.n)]
        return [a for a in acts if a not in masks]

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

    def to_str(self, val: int) -> str:
        return str(val)

    def get_default(self) -> int:
        return self._start

    def copy(self, **kwargs) -> "DiscreteSpace":
        keys = ["n", "start"]
        args = [kwargs.get(key, getattr(self, f"_{key}")) for key in keys]
        return DiscreteSpace(*args)

    def copy_value(self, v: int) -> int:
        return v

    def equal_val(self, v1: int, v2: int) -> bool:
        return v1 == v2

    def __eq__(self, o: "DiscreteSpace") -> bool:
        if not isinstance(o, DiscreteSpace):
            return False
        return self._n == o._n and self._start == o._start

    def __str__(self) -> str:
        return f"Discrete({self._n}, start={self._start})"

    # --- stack
    def create_stack_space(self, length: int):
        from srl.base.spaces.array_discrete import ArrayDiscreteSpace

        return ArrayDiscreteSpace(length, self._start, self._n + self._start - 1)

    def encode_stack(self, val: List[int]) -> List[int]:
        return val

    # --- utils
    def get_onehot(self, x: int) -> List[int]:
        onehot = [0] * self._n
        if x - self._start < 0:
            raise IndexError(f"Invalid value. {x} {self._start} {self._n}")
        onehot[x - self._start] = 1
        return onehot

    # --------------------------------------
    # spaces
    # --------------------------------------
    def get_encode_list(self):
        return [
            RLBaseTypes.DISCRETE,
            RLBaseTypes.ARRAY_DISCRETE,
            #
            RLBaseTypes.NP_ARRAY,
            RLBaseTypes.CONTINUOUS,
            RLBaseTypes.ARRAY_CONTINUOUS,
            RLBaseTypes.BOX,
            RLBaseTypes.TEXT,
            RLBaseTypes.MULTI,
        ]

    # --- DiscreteSpace
    def create_encode_space_DiscreteSpace(self):
        return DiscreteSpace(self._n)  # startは0

    def encode_to_space_DiscreteSpace(self, val: int) -> int:
        return val - self._start

    def decode_from_space_DiscreteSpace(self, val: int) -> int:
        return val + self._start

    # --- ArrayDiscreteSpace
    def create_encode_space_ArrayDiscreteSpace(self):
        from srl.base.spaces.array_discrete import ArrayDiscreteSpace

        return ArrayDiscreteSpace(1, 0, self._n - 1)

    def encode_to_space_ArrayDiscreteSpace(self, val: int) -> List[int]:
        return [val - self._start]

    def decode_from_space_ArrayDiscreteSpace(self, val: List[int]) -> int:
        return val[0] + self._start

    # --- ContinuousSpace
    def create_encode_space_ContinuousSpace(self):
        from srl.base.spaces.continuous import ContinuousSpace

        return ContinuousSpace(0, self._n - 1)

    def encode_to_space_ContinuousSpace(self, val: int) -> float:
        return float(val - self._start)

    def decode_from_space_ContinuousSpace(self, val: float) -> int:
        return int(round(val)) + self._start

    # --- ArrayContinuousSpace
    def create_encode_space_ArrayContinuousSpace(self):
        from srl.base.spaces.array_continuous import ArrayContinuousSpace

        return ArrayContinuousSpace(1, 0, self.n - 1)

    def encode_to_space_ArrayContinuousSpace(self, val: int) -> List[float]:
        return [float(val - self._start)]

    def decode_from_space_ArrayContinuousSpace(self, val: List[float]) -> int:
        return int(round(val[0])) + self._start

    # --- NpArray
    def create_encode_space_NpArraySpace(self, options: SpaceEncodeOptions):
        from srl.base.spaces.np_array import NpArraySpace

        dtype = options.cast_dtype if options.cast else np.uint
        return NpArraySpace(1, 0, self.n - 1, dtype, SpaceTypes.DISCRETE)

    def encode_to_space_NpArraySpace(self, val: int, to_space: SpaceBase) -> np.ndarray:
        return np.array([val - self._start], dtype=to_space.dtype)

    def decode_from_space_NpArraySpace(self, val: np.ndarray, from_space: SpaceBase) -> int:
        return int(round(float(val[0]))) + self._start

    # --- Box
    def create_encode_space_Box(self, options: SpaceEncodeOptions):
        from srl.base.spaces.box import BoxSpace

        dtype = options.cast_dtype if options.cast else np.uint
        return BoxSpace((1,), 0, self.n - 1, dtype, SpaceTypes.DISCRETE)

    def encode_to_space_Box(self, val: int, to_space: "BoxSpace") -> np.ndarray:
        return np.array([val - self._start], dtype=to_space.dtype)

    def decode_from_space_Box(self, x: np.ndarray, from_space: "BoxSpace") -> int:
        return int(round(x[0])) + self._start

    # --- TextSpace
    def create_encode_space_TextSpace(self):
        from srl.base.spaces.text import TextSpace

        # 特殊: e, +
        max_len = len(str(self.n - self.start))
        return TextSpace(max_len, min_length=1, charset="0123456789-")

    def encode_to_space_TextSpace(self, val: int) -> str:
        return str(val - self._start)

    def decode_from_space_TextSpace(self, val: str) -> int:
        return int(val) + self._start

    # --- Multi
    def create_encode_space_MultiSpace(self):
        from srl.base.spaces.multi import MultiSpace

        return MultiSpace([self.copy()])

    def encode_to_space_MultiSpace(self, val: int) -> list:
        return [val]

    def decode_from_space_MultiSpace(self, val: list) -> int:
        return val[0]
