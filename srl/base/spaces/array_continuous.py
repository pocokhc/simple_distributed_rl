import logging
import time
from typing import Any, List, Tuple, Union

import numpy as np

from srl.base.define import SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.space import SpaceBase

logger = logging.getLogger(__name__)


class ArrayContinuousSpace(SpaceBase[List[float]]):
    def __init__(
        self,
        size: int,
        low: Union[float, List[float], Tuple[float, ...], np.ndarray] = -np.inf,
        high: Union[float, List[float], Tuple[float, ...], np.ndarray] = np.inf,
        dtype=np.float32,
    ) -> None:
        self._size = size
        self._low: np.ndarray = np.full((size,), low) if np.isscalar(low) else np.asarray(low)
        self._high: np.ndarray = np.full((size,), high) if np.isscalar(high) else np.asarray(high)
        self._low = self._low.astype(dtype)
        self._high = self._high.astype(dtype)
        self._dtype = dtype

        assert len(self._low) == size
        assert len(self._high) == size
        assert self.low.shape == self.high.shape
        assert np.less_equal(self.low, self.high).all()

        self._is_inf = np.isinf(low).any() or np.isinf(high).any()
        self.division_tbl = None

    @property
    def size(self):
        return self._size

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

    def sample(self, mask: List[List[float]] = []) -> List[float]:
        if len(mask) > 0:
            logger.info(f"mask is not support: {mask}")
        if self._is_inf:
            # infの場合は正規分布に従う乱数
            return np.random.normal(size=(self._size,)).tolist()
        r = np.random.random_sample((self._size,))
        return (self._low + r * (self._high - self._low)).tolist()

    def sanitize(self, val: Any) -> List[float]:
        if isinstance(val, list):
            val = [float(v) for v in val]
        elif isinstance(val, tuple):
            val = [float(v) for v in val]
        elif isinstance(val, np.ndarray):
            val = val.tolist()
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

    def copy(self) -> "ArrayContinuousSpace":
        o = ArrayContinuousSpace(self._size, self._low, self._high)
        o.division_tbl = self.division_tbl
        return o

    def copy_value(self, v: List[float]) -> List[float]:
        return v[:]

    def __eq__(self, o: "ArrayContinuousSpace") -> bool:
        if not isinstance(o, ArrayContinuousSpace):
            return False
        return self._size == o._size and (self._low == o._low).all() and (self._high == o._high).all()

    def __str__(self) -> str:
        if self.division_tbl is None:
            s = ""
        else:
            s = f", division({self.int_size})"
        return f"ArrayContinuous({self._size}, range[{np.min(self.low)}, {np.max(self.high)}]){s}"

    # --- stack
    def create_stack_space(self, length: int):
        return ArrayContinuousSpace(
            length * self._size,
            length * self._low.tolist(),
            length * self._high.tolist(),
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
    # action discrete
    # --------------------------------------
    @property
    def int_size(self) -> int:
        assert self.division_tbl is not None, "Call 'create_division_tbl(division_num)' first"
        return len(self.division_tbl)

    def encode_to_int(self, val: List[float]) -> int:
        assert self.division_tbl is not None, "Call 'create_division_tbl(division_num)' first"
        d = np.sum(np.abs(self.division_tbl - val), axis=1)
        return int(np.argmin(d))

    def decode_from_int(self, val: int) -> List[float]:
        if self.division_tbl is None:
            return [float(val) for _ in range(self._size)]
        else:
            return self.division_tbl[val].tolist()

    # --------------------------------------
    # observation discrete
    # --------------------------------------
    @property
    def list_int_size(self) -> int:
        if self.division_tbl is None:
            return self._size
        else:
            return 1

    @property
    def list_int_low(self) -> List[int]:
        if self.division_tbl is None:
            return [int(n) for n in np.round(self._low).tolist()]
        else:
            return [0]

    @property
    def list_int_high(self) -> List[int]:
        if self.division_tbl is None:
            return [int(n) for n in np.round(self._high).tolist()]
        else:
            return [self.int_size]

    def encode_to_list_int(self, val: List[float]) -> List[int]:
        if self.division_tbl is None:
            return [int(round(v)) for v in val]
        else:
            return [self.encode_to_int(val)]

    def decode_from_list_int(self, val: List[int]) -> List[float]:
        if self.division_tbl is None:
            return [float(v) for v in val]
        else:
            return self.division_tbl[val[0]].tolist()

    # --------------------------------------
    # action continuous
    # --------------------------------------
    @property
    def list_float_size(self) -> int:
        return self._size

    @property
    def list_float_low(self) -> List[float]:
        return self._low.tolist()

    @property
    def list_float_high(self) -> List[float]:
        return self._high.tolist()

    def encode_to_list_float(self, val: List[float]) -> List[float]:
        return val

    def decode_from_list_float(self, val: List[float]) -> List[float]:
        return val

    # --------------------------------------
    # observation continuous, image
    # --------------------------------------
    @property
    def np_shape(self) -> Tuple[int, ...]:
        return (self._size,)

    @property
    def np_low(self) -> np.ndarray:
        return self._low

    @property
    def np_high(self) -> np.ndarray:
        return self._high

    def encode_to_np(self, val: List[float], dtype) -> np.ndarray:
        return np.array(val, dtype=dtype)

    def decode_from_np(self, val: np.ndarray) -> List[float]:
        return val.tolist()

    # --------------------------------------
    # spaces
    # --------------------------------------
    def create_encode_space(self, space_name: str) -> SpaceBase:
        from srl.base.spaces.array_discrete import ArrayDiscreteSpace
        from srl.base.spaces.box import BoxSpace
        from srl.base.spaces.discrete import DiscreteSpace

        if space_name == "":
            return self.copy()
        elif space_name == "DiscreteSpace":
            return DiscreteSpace(self.int_size)
        elif space_name == "ArrayDiscreteSpace":
            return ArrayDiscreteSpace(self.list_int_size, self.list_int_low, self.list_int_high)
        elif space_name == "ContinuousSpace":
            raise NotSupportedError()
        elif space_name == "ArrayContinuousSpace":
            return ArrayContinuousSpace(self.list_float_size, self.list_float_low, self.list_float_high)
        elif space_name == "BoxSpace":
            return BoxSpace(self.np_shape, self.np_low, self.np_high, self._dtype)
        elif space_name == "BoxSpace_float":
            return BoxSpace(self.np_shape, self.np_low, self.np_high, np.float32)
        elif space_name == "TextSpace":
            raise NotSupportedError()
        raise NotImplementedError(space_name)

    def encode_to_space(self, val: List[float], space: SpaceBase) -> Any:
        from srl.base.spaces.array_discrete import ArrayDiscreteSpace
        from srl.base.spaces.box import BoxSpace
        from srl.base.spaces.continuous import ContinuousSpace
        from srl.base.spaces.discrete import DiscreteSpace
        from srl.base.spaces.multi import MultiSpace
        from srl.base.spaces.text import TextSpace

        if isinstance(space, DiscreteSpace):
            assert self.division_tbl is not None, "Call 'create_division_tbl(division_num)' first"
            d = np.sum(np.abs(self.division_tbl - val), axis=1)
            return int(np.argmin(d))
        elif isinstance(space, ArrayDiscreteSpace):
            if self.division_tbl is None:
                return [int(round(v)) for v in val]
            else:
                return [self.encode_to_int(val)]
        elif isinstance(space, ContinuousSpace):
            raise NotImplementedError()
        elif isinstance(space, ArrayContinuousSpace):
            return val
        elif isinstance(space, BoxSpace):
            return np.array(val, space.dtype)
        elif isinstance(space, TextSpace):
            return ",".join([str(v) for v in val])
        elif isinstance(space, MultiSpace):
            return val
        raise NotImplementedError()

    def decode_from_space(self, val: Any, space: SpaceBase) -> List[float]:
        from srl.base.spaces.array_discrete import ArrayDiscreteSpace
        from srl.base.spaces.box import BoxSpace
        from srl.base.spaces.continuous import ContinuousSpace
        from srl.base.spaces.discrete import DiscreteSpace
        from srl.base.spaces.multi import MultiSpace
        from srl.base.spaces.text import TextSpace

        if isinstance(space, DiscreteSpace):
            if self.division_tbl is None:
                return [float(val) for _ in range(self._size)]
            else:
                return self.division_tbl[val].tolist()
        elif isinstance(space, ArrayDiscreteSpace):
            if self.division_tbl is None:
                return [float(v) for v in val]
            else:
                return self.division_tbl[val[0]].tolist()
        elif isinstance(space, ContinuousSpace):
            raise NotImplementedError()
        elif isinstance(space, ArrayContinuousSpace):
            return val
        elif isinstance(space, BoxSpace):
            return val.tolist()
        elif isinstance(space, TextSpace):
            return [float(v) for v in val.split(",")]
        elif isinstance(space, MultiSpace):
            return val
        raise NotImplementedError()
