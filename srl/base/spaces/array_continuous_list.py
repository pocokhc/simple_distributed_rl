import logging
import random
import time
from typing import Any, List, Tuple, Union

import numpy as np

from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.space import SpaceBase

logger = logging.getLogger(__name__)


class ArrayContinuousListSpace(SpaceBase[List[float]]):
    def __init__(
        self,
        size: int,
        low: Union[float, List[float], Tuple[float, ...], np.ndarray] = float("-inf"),
        high: Union[float, List[float], Tuple[float, ...], np.ndarray] = float("inf"),
    ) -> None:
        if isinstance(low, (float, int)):
            self._low: list[float] = [float(low)] * size
        else:
            self._low = list(np.array(low, dtype=float).flatten())

        if isinstance(high, (float, int)):
            self._high: list[float] = [float(high)] * size
        else:
            self._high = list(np.array(high, dtype=float).flatten())

        self._size = size

        assert size > 0
        assert len(self._low) == size
        assert len(self._high) == size
        assert np.less_equal(self.low, self.high).all()

        self._is_inf = np.isinf(low).any() or np.isinf(high).any()
        self.division_tbl = None

    @property
    def size(self) -> int:
        return self._size

    @property
    def low(self) -> List[float]:
        return self._low

    @property
    def high(self) -> List[float]:
        return self._high

    def rescale_from(self, x: List[float], src_low: float, src_high: float) -> List[float]:
        assert src_low < src_high
        x = [
            ((x[i] - src_low) / (src_high - src_low)) * (self._high[i] - self._low[i]) + self._low[i]
            for i in range(len(x))  #
        ]
        return x

    # ----------------------------------------

    @property
    def stype(self) -> SpaceTypes:
        return SpaceTypes.CONTINUOUS

    @property
    def dtype(self):
        return np.float32

    @property
    def name(self) -> str:
        return "ArrayContinuousList"

    def sample(self, mask: List[List[float]] = []) -> List[float]:
        if len(mask) > 0:
            logger.info(f"mask is not support: {mask}")
        if self._is_inf:
            # infの場合は正規分布
            return [random.gauss(mu=0.0, sigma=1.0) for _ in range(self._size)]
        # 一様分布
        return [
            random.random() * (self._high[i] - self._low[i]) + self._low[i]
            for i in range(self._size)  #
        ]

    def sanitize(self, val: Any) -> List[float]:
        if isinstance(val, np.ndarray):
            val = val.tolist()
        if isinstance(val, list):
            val = [float(v) for v in val]
        elif isinstance(val, tuple):
            val = [float(v) for v in val]
        else:
            val = [float(val) for _ in range(self._size)]
        for i in range(self._size):
            if val[i] < self._low[i]:
                val[i] = float(self._low[i])
            elif val[i] > self._high[i]:
                val[i] = float(self._high[i])
        return val

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

    def to_str(self, val: List[float]) -> str:
        return ",".join([str(int(v) if v.is_integer() else v) for v in val])

    def get_default(self) -> List[float]:
        return [0 if self._low[i] <= 0 <= self._high[i] else self._low[i] for i in range(self._size)]

    def copy(self, **kwargs) -> "ArrayContinuousListSpace":
        keys = ["size", "low", "high"]
        args = [kwargs.get(key, getattr(self, f"_{key}")) for key in keys]
        o = ArrayContinuousListSpace(*args)
        o.division_tbl = self.division_tbl
        return o

    def copy_value(self, v: List[float]) -> List[float]:
        return v[:]

    def __eq__(self, o: "ArrayContinuousListSpace") -> bool:
        if not isinstance(o, ArrayContinuousListSpace):
            return False
        if self._size != o._size:
            return False
        for i in range(self._size):
            if self._low[i] != o._low[i]:
                return False
            if self._high[i] != o._high[i]:
                return False
        return True

    def __str__(self) -> str:
        if self.division_tbl is None:
            s = ""
        else:
            s = f", division({len(self.division_tbl)})"
        return f"ArrayContinuousList({self._size}, range[{min(self._low)}, {max(self._high)}]){s}"

    # --- stack
    def create_stack_space(self, length: int):
        return ArrayContinuousListSpace(
            length * self._size,
            length * self._low,
            length * self._high,
        )

    def encode_stack(self, val: List[List[float]]) -> List[float]:
        return [e for sublist in val for e in sublist]

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

        # 各要素は分割後のデカルト積のサイズが division_num になるように分割
        # ただし、各要素の最低は2
        # division_num = division_num_one ** size
        division_num_one = round(division_num ** (1 / self._size))
        if division_num_one < 2:
            division_num_one = 2

        import itertools

        t0 = time.time()
        div_list = []
        for i in range(self._size):
            low = self._low[i]
            high = self._high[i]
            diff = (high - low) / (division_num_one - 1)
            div_list.append([float(low + diff * j) for j in range(division_num_one)])

        # --- 多いと時間がかかるので切り上げる
        byte_size = -1
        div_prods = []
        for prod in itertools.product(*div_list):
            if byte_size == -1:
                byte_size = len(prod) * 4
            div_prods.append(prod)
            if len(div_prods) >= max_size:
                break
            if len(div_prods) * byte_size >= max_byte:
                break
        self.division_tbl = np.array(div_prods)
        n = len(self.division_tbl)

        logger.info(f"created division: size={n}, create time={time.time() - t0:.3f}s")

    # --------------------------------------
    # spaces
    # --------------------------------------
    def get_encode_type_list(self):
        priority_list = [
            RLBaseTypes.ARRAY_CONTINUOUS_LIST,
            RLBaseTypes.ARRAY_CONTINUOUS,
            RLBaseTypes.BOX,
            RLBaseTypes.ARRAY_DISCRETE,
        ]
        exclude_list = [RLBaseTypes.CONTINUOUS]
        return priority_list, exclude_list

    # --- DiscreteSpace
    def create_encode_space_DiscreteSpace(self):
        from srl.base.spaces.discrete import DiscreteSpace

        assert self.division_tbl is not None, "Call 'create_division_tbl(division_num)' first"
        return DiscreteSpace(len(self.division_tbl))  # startは0

    def encode_to_space_DiscreteSpace(self, val: List[float]) -> int:
        assert self.division_tbl is not None, "Call 'create_division_tbl(division_num)' first"
        d = np.sum(np.abs(self.division_tbl - val), axis=1)
        return int(np.argmin(d))

    def decode_from_space_DiscreteSpace(self, val: int) -> List[float]:
        if self.division_tbl is None:
            # not comming
            return [float(val) for _ in range(self._size)]
        else:
            return self.division_tbl[val].tolist()

    # --- ArrayDiscreteSpace
    def create_encode_space_ArrayDiscreteSpace(self):
        from srl.base.spaces.array_discrete import ArrayDiscreteSpace

        if self.division_tbl is None:
            return ArrayDiscreteSpace(
                size=self._size,
                low=[int(round(n)) for n in self._low],
                high=[int(round(n)) for n in self._high],
            )
        else:
            return ArrayDiscreteSpace(1, 0, len(self.division_tbl))

    def encode_to_space_ArrayDiscreteSpace(self, val: List[float]) -> List[int]:
        if self.division_tbl is None:
            return [int(round(v)) for v in val]
        else:
            return [self.encode_to_space_DiscreteSpace(val)]

    def decode_from_space_ArrayDiscreteSpace(self, val: List[int]) -> List[float]:
        if self.division_tbl is None:
            return [float(v) for v in val]
        else:
            return self.division_tbl[val[0]].tolist()

    # --- ContinuousSpace
    def create_encode_space_ContinuousSpace(self):
        raise NotSupportedError()

    def encode_to_space_ContinuousSpace(self, val: List[float]) -> float:
        raise NotSupportedError()

    def decode_from_space_ContinuousSpace(self, val: float) -> List[float]:
        raise NotSupportedError()

    # --- ArrayContinuousSpace
    def create_encode_space_ArrayContinuousListSpace(self):
        from srl.base.spaces.array_continuous_list import ArrayContinuousListSpace

        return ArrayContinuousListSpace(self._size, self._low, self._high)

    def encode_to_space_ArrayContinuousListSpace(self, val: List[float]) -> List[float]:
        return val

    def decode_from_space_ArrayContinuousListSpace(self, val: List[float]) -> List[float]:
        return val

    # --- np
    def create_encode_space_ArrayContinuousSpace(self, np_dtype):
        from srl.base.spaces.array_continuous import ArrayContinuousSpace

        return ArrayContinuousSpace(self._size, self._low, self._high, np_dtype)

    def encode_to_space_ArrayContinuousSpace(self, val: List[float], space) -> np.ndarray:
        return np.asarray(val, dtype=space.dtype)

    def decode_from_space_ArrayContinuousSpace(self, val: np.ndarray) -> List[float]:
        return val.tolist()

    # --- Box
    def create_encode_space_Box(self, space_type: RLBaseTypes, np_dtype):
        from srl.base.spaces.box import BoxSpace

        if space_type == RLBaseTypes.GRAY_2ch:
            if self._size == 1:
                return BoxSpace((1, 1), self._low, self._high, dtype=np_dtype, stype=SpaceTypes.GRAY_2ch)
            else:
                logger.warning("Not defined. Converts CONTINUOUS.")
        elif space_type == RLBaseTypes.GRAY_3ch:
            if self._size == 1:
                return BoxSpace((1, 1, 1), self._low, self._high, dtype=np_dtype, stype=SpaceTypes.GRAY_3ch)
            else:
                logger.warning("Not defined. Converts CONTINUOUS.")
        elif space_type == RLBaseTypes.COLOR:
            if self._size == 1 or self._size == 3:
                return BoxSpace((1, 1, 3), self._low, self._high, dtype=np_dtype, stype=SpaceTypes.COLOR)
            else:
                logger.warning("Not defined. Converts CONTINUOUS.")
        elif space_type == RLBaseTypes.IMAGE:
            return BoxSpace((1, 1, self._size), self._low, self._high, dtype=np_dtype, stype=SpaceTypes.IMAGE)
        return BoxSpace((self._size,), self._low, self._high, dtype=np_dtype, stype=SpaceTypes.CONTINUOUS)

    def encode_to_space_Box(self, val: List[float], space) -> np.ndarray:
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

    def decode_from_space_Box(self, val: np.ndarray, space) -> List[float]:
        if space.stype == SpaceTypes.GRAY_2ch:
            return val.reshape(-1).tolist()
        elif space.stype == SpaceTypes.GRAY_3ch:
            return val.reshape(-1).tolist()
        elif space.stype == SpaceTypes.COLOR:
            if self._size == 1:
                return val[0, 0, 0].tolist()
            else:
                return val.reshape(-1).tolist()
        elif space.stype == SpaceTypes.IMAGE:
            return val.reshape(-1).tolist()
        return val.tolist()

    # --- TextSpace
    def create_encode_space_TextSpace(self):
        from srl.base.spaces.text import TextSpace

        return TextSpace(min_length=1, charset="0123456789-.,")

    def encode_to_space_TextSpace(self, val: List[float]) -> str:
        return ",".join([str(v) for v in val])

    def decode_from_space_TextSpace(self, val: str) -> List[float]:
        return [float(v) for v in val.split(",")]
