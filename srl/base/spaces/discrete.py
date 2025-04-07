import logging
import random
from typing import Any, List, Tuple

import numpy as np

from srl.base.define import SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.space import SpaceBase

logger = logging.getLogger(__name__)


class DiscreteSpace(SpaceBase[int]):
    def __init__(self, n: int, start: int = 0) -> None:
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
    # action discrete
    # --------------------------------------
    @property
    def int_size(self) -> int:
        return self._n

    def encode_to_int(self, val: int) -> int:
        return val - self._start

    def decode_from_int(self, val: int) -> int:
        return val + self._start

    # --------------------------------------
    # observation discrete
    # --------------------------------------
    @property
    def list_int_size(self) -> int:
        return 1

    @property
    def list_int_low(self) -> List[int]:
        return [0]

    @property
    def list_int_high(self) -> List[int]:
        return [self.n - 1]

    def encode_to_list_int(self, val: int) -> List[int]:
        return [val - self._start]

    def decode_from_list_int(self, val: List[int]) -> int:
        return int(round(val[0])) + self._start

    # --------------------------------------
    # action continuous
    # --------------------------------------
    @property
    def list_float_size(self) -> int:
        return 1

    @property
    def list_float_low(self) -> List[float]:
        return [0]

    @property
    def list_float_high(self) -> List[float]:
        return [self.n - 1]

    def encode_to_list_float(self, val: int) -> List[float]:
        return [float(val - self._start)]

    def decode_from_list_float(self, val: List[float]) -> int:
        return int(round(val[0])) + self._start

    # --------------------------------------
    # observation continuous, image
    # --------------------------------------
    @property
    def np_shape(self) -> Tuple[int, ...]:
        return (1,)

    @property
    def np_low(self) -> np.ndarray:
        return np.array([0])

    @property
    def np_high(self) -> np.ndarray:
        return np.array([self.n - 1])

    def encode_to_np(self, val: int, dtype) -> np.ndarray:
        return np.array([val - self._start], dtype=dtype)

    def decode_from_np(self, val: np.ndarray) -> int:
        return int(round(val[0])) + self._start

    # --------------------------------------
    # spaces
    # --------------------------------------
    def create_encode_space(self, space_name: str) -> SpaceBase:
        from srl.base.spaces.array_continuous import ArrayContinuousSpace
        from srl.base.spaces.array_discrete import ArrayDiscreteSpace
        from srl.base.spaces.box import BoxSpace
        from srl.base.spaces.continuous import ContinuousSpace

        if space_name == "":
            return DiscreteSpace(self.int_size)
        elif space_name == "DiscreteSpace":
            return DiscreteSpace(self.int_size)
        elif space_name == "ArrayDiscreteSpace":
            return ArrayDiscreteSpace(self.list_int_size, self.list_int_low, self.list_int_high)
        elif space_name == "ContinuousSpace":
            return ContinuousSpace(0, self._n - 1)
        elif space_name == "ArrayContinuousSpace":
            return ArrayContinuousSpace(self.list_float_size, self.list_float_low, self.list_float_high)
        elif space_name == "BoxSpace":
            return BoxSpace((1,), 0, self.n - 1, np.int64, SpaceTypes.DISCRETE)
        elif space_name == "BoxSpace_float":
            return BoxSpace((1,), 0, self.n - 1, np.float32, SpaceTypes.DISCRETE)
        # elif stype == EncodeSpaceCreateTypes.GRAY_2ch:
        #    return BoxSpace((1, 1), 0, self.n - 1, np.int64, SpaceTypes.GRAY_2ch)
        # elif stype == EncodeSpaceCreateTypes.GRAY_3ch:
        #    return BoxSpace((1, 1, 1), 0, self.n - 1, np.int64, SpaceTypes.GRAY_3ch)
        # elif stype == EncodeSpaceCreateTypes.COLOR:
        #    return BoxSpace((1, 1, 3), 0, self.n - 1, np.int64, SpaceTypes.COLOR)
        # elif stype == EncodeSpaceCreateTypes.IMAGE:
        #    return BoxSpace((1, 1, 1), 0, self.n - 1, np.int64, SpaceTypes.IMAGE)
        elif space_name == "TextSpace":
            raise NotSupportedError()
        raise NotImplementedError(space_name)

    def encode_to_space(self, val: int, space: SpaceBase) -> Any:
        from srl.base.spaces.array_continuous import ArrayContinuousSpace
        from srl.base.spaces.array_discrete import ArrayDiscreteSpace
        from srl.base.spaces.box import BoxSpace
        from srl.base.spaces.continuous import ContinuousSpace
        from srl.base.spaces.multi import MultiSpace
        from srl.base.spaces.text import TextSpace

        if isinstance(space, DiscreteSpace):
            return val - self._start
        elif isinstance(space, ArrayDiscreteSpace):
            return [val - self._start]
        elif isinstance(space, ContinuousSpace):
            return float(val - self._start)
        elif isinstance(space, ArrayContinuousSpace):
            return [float(val - self._start)]
        elif isinstance(space, BoxSpace):
            v = np.array([val - self._start], space.dtype)
            if space.stype == SpaceTypes.GRAY_2ch:
                v = v.reshape((1, 1))
            elif space.stype == SpaceTypes.GRAY_3ch:
                v = v.reshape((1, 1, 1))
            elif space.stype == SpaceTypes.COLOR:
                v = np.full((1, 1, 3), v[0], space.dtype)
            elif space.stype == SpaceTypes.IMAGE:
                v = v.reshape((1, 1, 1))
            return v
        elif isinstance(space, TextSpace):
            return str(val - self._start)
        elif isinstance(space, MultiSpace):
            return val
        raise NotImplementedError()

    def decode_from_space(self, val: Any, space: SpaceBase) -> int:
        from srl.base.spaces.array_continuous import ArrayContinuousSpace
        from srl.base.spaces.array_discrete import ArrayDiscreteSpace
        from srl.base.spaces.box import BoxSpace
        from srl.base.spaces.continuous import ContinuousSpace
        from srl.base.spaces.multi import MultiSpace
        from srl.base.spaces.text import TextSpace

        if isinstance(space, DiscreteSpace):
            return val + self._start
        elif isinstance(space, ArrayDiscreteSpace):
            return val[0] + self._start
        elif isinstance(space, ContinuousSpace):
            return int(round(val)) + self._start
        elif isinstance(space, ArrayContinuousSpace):
            return int(round(val[0])) + self._start
        elif isinstance(space, BoxSpace):
            if space.stype == SpaceTypes.GRAY_2ch:
                val = val.reshape(-1)
            elif space.stype == SpaceTypes.GRAY_3ch:
                val = val.reshape(-1)
            elif space.stype == SpaceTypes.COLOR:
                val = [np.mean(val)]
            elif space.stype == SpaceTypes.IMAGE:
                val = val.reshape(-1)
            return int(round(val[0])) + self._start
        elif isinstance(space, TextSpace):
            return int(val) + self._start
        elif isinstance(space, MultiSpace):
            return val
        raise NotImplementedError()
