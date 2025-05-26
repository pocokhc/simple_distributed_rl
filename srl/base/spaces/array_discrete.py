import logging
import random
import time
from typing import Any, List, Union

import numpy as np

from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.space import SpaceBase

logger = logging.getLogger(__name__)


class ArrayDiscreteSpace(SpaceBase[List[int]]):
    def __init__(
        self,
        size: int,
        low: Union[int, List[int]],
        high: Union[int, List[int]],
    ) -> None:
        assert size > 0
        self._size = size

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

    # ----------------------------------------
    @property
    def stype(self) -> SpaceTypes:
        return SpaceTypes.DISCRETE

    @property
    def dtype(self):
        return np.int64

    @property
    def name(self) -> str:
        return "ArrayDiscrete"

    def sample(self, mask: List[List[int]] = []) -> List[int]:
        if len(mask) > 0:
            self._create_tbl()
            assert self.encode_tbl is not None
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

    def get_valid_actions(self, masks: List[List[int]] = []) -> List[List[int]]:
        self._create_tbl()
        assert self.encode_tbl is not None
        valid_acts = []
        for a in [k for k in self.encode_tbl.keys()]:
            f = True
            for m in masks:
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

    def copy(self, **kwargs) -> "ArrayDiscreteSpace":
        keys = ["size", "low", "high"]
        args = [kwargs.get(key, getattr(self, f"_{key}")) for key in keys]
        o = ArrayDiscreteSpace(*args)
        o.decode_tbl = self.decode_tbl
        o.encode_tbl = self.encode_tbl
        return o

    def copy_value(self, v: List[int]) -> List[int]:
        return v[:]

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

    # --- utils
    def get_onehot(self, x: List[int]) -> List[List[int]]:
        onehot = []
        for i, val in enumerate(x):
            if val < self._low[i] or val > self._high[i]:
                raise ValueError(f"Value {val} at index {i} is out of bounds [{self._low[i]}, {self._high[i]}].")
            vector = [0] * (self._high[i] - self._low[i] + 1)
            vector[val - self._low[i]] = 1
            onehot.append(vector)
        return onehot

    # --------------------------------------
    # create_tbl
    # --------------------------------------
    def _create_tbl(
        self,
        max_size: int = 100_000,
        max_byte: int = 1024 * 1024 * 1024,
    ) -> None:
        if self.decode_tbl is not None:
            return

        import itertools

        logger.info("create table start")
        t0 = time.time()
        arr_list = [[a for a in range(self.low[i], self.high[i] + 1)] for i in range(self._size)]

        # --- 多いと時間がかかるので切り上げる
        byte_size = -1
        self.decode_tbl = []
        for prod in itertools.product(*arr_list):
            if byte_size == -1:
                byte_size = len(prod) * 4
            self.decode_tbl.append(prod)
            if len(self.decode_tbl) >= max_size:
                break
            if len(self.decode_tbl) * byte_size >= max_byte:
                break

        self.encode_tbl = {}
        for i, v in enumerate(self.decode_tbl):
            self.encode_tbl[v] = i
        logger.info(f"create table: size={len(self.decode_tbl)}, create time: {time.time() - t0:.3f}s")

    # --------------------------------------
    # spaces
    # --------------------------------------
    def get_encode_type_list(self):
        priority_list = [
            RLBaseTypes.ARRAY_DISCRETE,
            RLBaseTypes.DISCRETE,
            RLBaseTypes.ARRAY_CONTINUOUS_LIST,
            RLBaseTypes.ARRAY_CONTINUOUS,
            RLBaseTypes.BOX,
        ]
        exclude_list = [RLBaseTypes.CONTINUOUS]
        return priority_list, exclude_list

    # --- DiscreteSpace
    def create_encode_space_DiscreteSpace(self):
        from srl.base.spaces.discrete import DiscreteSpace

        self._create_tbl()
        assert self.decode_tbl is not None
        return DiscreteSpace(len(self.decode_tbl))  # startは0

    def encode_to_space_DiscreteSpace(self, val: List[int]) -> int:
        self._create_tbl()
        assert self.encode_tbl is not None
        return self.encode_tbl[tuple(val)]

    def decode_from_space_DiscreteSpace(self, val: int) -> List[int]:
        self._create_tbl()
        assert self.decode_tbl is not None
        return list(self.decode_tbl[val])

    # --- ArrayDiscreteSpace
    def create_encode_space_ArrayDiscreteSpace(self):
        from srl.base.spaces.array_discrete import ArrayDiscreteSpace

        return ArrayDiscreteSpace(self._size, self._low, self._high)

    def encode_to_space_ArrayDiscreteSpace(self, val: List[int]) -> List[int]:
        return val

    def decode_from_space_ArrayDiscreteSpace(self, val: List[int]) -> List[int]:
        return val

    # --- ContinuousSpace
    def create_encode_space_ContinuousSpace(self):
        raise NotSupportedError()

    def encode_to_space_ContinuousSpace(self, val: List[int]) -> float:
        raise NotSupportedError()

    def decode_from_space_ContinuousSpace(self, val: float) -> List[int]:
        raise NotSupportedError()

    # --- ArrayContinuousSpace
    def create_encode_space_ArrayContinuousListSpace(self):
        from srl.base.spaces.array_continuous_list import ArrayContinuousListSpace

        return ArrayContinuousListSpace(
            self._size,
            [float(n) for n in self._low],
            [float(n) for n in self._high],
        )

    def encode_to_space_ArrayContinuousListSpace(self, val: List[int]) -> List[float]:
        return [float(v) for v in val]

    def decode_from_space_ArrayContinuousListSpace(self, val: List[float]) -> List[int]:
        return [int(round(v)) for v in val]

    # --- ArrayContinuousNpSpace
    def create_encode_space_ArrayContinuousSpace(self, np_dtype):
        from srl.base.spaces.array_continuous import ArrayContinuousSpace

        return ArrayContinuousSpace(
            self._size,
            np.array(self._low, dtype=np_dtype),
            np.array(self._high, dtype=np_dtype),
            dtype=np_dtype,
        )

    def encode_to_space_ArrayContinuousSpace(self, val: List[int], space) -> np.ndarray:
        return np.asarray(val, dtype=space.dtype)

    def decode_from_space_ArrayContinuousSpace(self, val: np.ndarray) -> List[int]:
        return [int(round(v)) for v in val.tolist()]

    # --- Box
    def create_encode_space_Box(self, space_type: RLBaseTypes, np_dtype):
        from srl.base.spaces.box import BoxSpace

        low = np.array(self._low, dtype=np_dtype)
        high = np.array(self._high, dtype=np_dtype)
        if space_type == RLBaseTypes.GRAY_2ch:
            if self._size == 1:
                return BoxSpace((1, 1), low, high, dtype=np_dtype, stype=SpaceTypes.GRAY_2ch)
            else:
                logger.warning("Not defined. Converts CONTINUOUS.")
        elif space_type == RLBaseTypes.GRAY_3ch:
            if self._size == 1:
                return BoxSpace((1, 1, 1), low, high, dtype=np_dtype, stype=SpaceTypes.GRAY_2ch)
            else:
                logger.warning("Not defined. Converts CONTINUOUS.")
        elif space_type == RLBaseTypes.COLOR:
            if self._size == 1 or self._size == 3:
                return BoxSpace((1, 1, 3), low, high, dtype=np_dtype, stype=SpaceTypes.COLOR)
            else:
                logger.warning("Not defined. Converts CONTINUOUS.")
        elif space_type == RLBaseTypes.IMAGE:
            return BoxSpace((1, 1, self._size), low, high, dtype=np_dtype, stype=SpaceTypes.IMAGE)
        return BoxSpace((self._size,), low, high, dtype=np_dtype, stype=SpaceTypes.CONTINUOUS)

    def encode_to_space_Box(self, val: List[int], space) -> np.ndarray:
        if space.stype == SpaceTypes.GRAY_2ch:
            return np.asarray([val], dtype=space.dtype)
        elif space.stype == SpaceTypes.GRAY_3ch:
            return np.asarray([[val]], dtype=space.dtype)
        elif space.stype == SpaceTypes.COLOR:
            if self._size == 1:
                return np.full((1, 1, 3), val[0], dtype=space.dtype)
            else:
                return np.asarray([[val]], dtype=space.dtype)
        elif space.stype == SpaceTypes.IMAGE:
            return np.asarray([[val]], dtype=space.dtype)
        return np.asarray(val, dtype=space.dtype)

    def decode_from_space_Box(self, val: np.ndarray, space) -> List[int]:
        if space.stype == SpaceTypes.GRAY_2ch:
            arr = val.reshape(-1).tolist()
        elif space.stype == SpaceTypes.GRAY_3ch:
            arr = val.reshape(-1).tolist()
        elif space.stype == SpaceTypes.COLOR:
            if self._size == 1:
                arr = val[0, 0, 0].tolist()
            else:
                arr = val.reshape(-1).tolist()
        elif space.stype == SpaceTypes.IMAGE:
            arr = val.reshape(-1).tolist()
        else:
            arr = val.tolist()
        return [int(round(a)) for a in arr]

    # --- TextSpace
    def create_encode_space_TextSpace(self):
        from srl.base.spaces.text import TextSpace

        return TextSpace(
            min_length=1,
            charset="0123456789,",
        )

    def encode_to_space_TextSpace(self, val: List[int]) -> str:
        return ",".join([str(v) for v in val])

    def decode_from_space_TextSpace(self, val: str) -> List[int]:
        return [int(v) for v in val.split(",")]
