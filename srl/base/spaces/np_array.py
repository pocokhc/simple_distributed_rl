import logging
import time
from typing import Any, List, Tuple, Union

import numpy as np

from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.space import SpaceBase, SpaceEncodeOptions

logger = logging.getLogger(__name__)


class NpArraySpace(SpaceBase[np.ndarray]):
    def __init__(
        self,
        size: int,
        low: Union[float, List[float], Tuple[float, ...], np.ndarray] = -np.inf,
        high: Union[float, List[float], Tuple[float, ...], np.ndarray] = np.inf,
        dtype: Any = np.float32,
        stype: SpaceTypes = SpaceTypes.UNKNOWN,
    ) -> None:
        super().__init__()
        self._size = size
        self._low: np.ndarray = np.full((size,), low) if np.isscalar(low) else np.asarray(low)
        self._high: np.ndarray = np.full((size,), high) if np.isscalar(high) else np.asarray(high)
        self._low = self._low.astype(dtype)
        self._high = self._high.astype(dtype)
        self._dtype = dtype
        if stype == SpaceTypes.UNKNOWN:
            self._stype = SpaceTypes.DISCRETE if "int" in str(dtype) else SpaceTypes.CONTINUOUS
        else:
            self._stype = stype

        assert size > 0
        assert len(self._low) == size
        assert len(self._high) == size
        assert self.low.shape == self.high.shape
        assert np.less_equal(self.low, self.high).all()

        self._is_inf = np.isinf(low).any() or np.isinf(high).any()
        self.division_tbl = None

    @property
    def size(self) -> int:
        return self._size

    @property
    def shape(self) -> tuple:
        return (self._size,)

    @property
    def low(self) -> np.ndarray:
        return self._low

    @property
    def high(self) -> np.ndarray:
        return self._high

    def rescale_from(self, x: np.ndarray, src_low: float, src_high: float) -> np.ndarray:
        assert src_low < src_high
        x = ((x - src_low) / (src_high - src_low)) * (self._high - self._low) + self._low
        return x

    # ----------------------------------------

    @property
    def stype(self) -> SpaceTypes:
        return self._stype

    @property
    def dtype(self):
        return self._dtype

    @property
    def name(self) -> str:
        return "NpArray"

    def sample(self, mask: List[List[float]] = []) -> np.ndarray:
        if len(mask) > 0:
            logger.info(f"mask is not support: {mask}")
        if self._is_inf:
            # infの場合は正規分布
            return np.random.normal(size=(self._size,))
        # 一様分布
        r = np.random.random_sample((self._size,))
        return self._low + r * (self._high - self._low)

    def sanitize(self, val: Any) -> np.ndarray:
        val = np.asarray(val, dtype=self._dtype).flatten()
        return np.clip(val, self._low, self._high)

    def check_val(self, val: Any) -> bool:
        if not isinstance(val, np.ndarray):
            return False
        if val.shape != (self._size,):
            return False
        if np.isnan(val).any():
            return False
        for i in range(self._size):
            if val[i] < self.low[i]:
                return False
            if val[i] > self.high[i]:
                return False
        return True

    def to_str(self, val: np.ndarray) -> str:
        return ",".join([str(int(v) if v.is_integer() else v) for v in val.tolist()])

    def get_default(self) -> np.ndarray:
        return np.where((self._low < 0) & (0 < self.high), 0, self._low)

    def copy(self, **kwargs) -> "NpArraySpace":
        keys = ["size", "low", "high", "dtype", "stype"]
        args = [kwargs.get(key, getattr(self, f"_{key}")) for key in keys]
        o = NpArraySpace(*args)
        o.division_tbl = self.division_tbl
        return o

    def copy_value(self, v: np.ndarray) -> np.ndarray:
        return v.copy()

    def equal_val(self, v1: np.ndarray, v2: np.ndarray, rtol=1e-9, atol=1e-12) -> bool:
        if v1.shape != v2.shape:
            return False
        if np.issubdtype(v1.dtype, np.floating) or np.issubdtype(v2.dtype, np.floating):
            return np.allclose(v1, v2, rtol=rtol, atol=atol)
        return np.array_equal(v1, v2)

    def __eq__(self, o: "NpArraySpace") -> bool:
        if not isinstance(o, NpArraySpace):
            return False
        return (
            self._size == o._size
            and (self._low == o._low).all()  #
            and (self._high == o._high).all()
            and (self._dtype == o._dtype)
            and (self._stype == o._stype)
        )

    def __str__(self) -> str:
        if self.division_tbl is None:
            s = ""
        else:
            s = f", division({len(self.division_tbl)})"
        return f"NpArray[{self.stype.name}]({self._size}, range[{np.min(self.low)}, {np.max(self.high)}], {self._dtype}){s}"

    # --- stack
    def create_stack_space(self, length: int):
        return NpArraySpace(
            length * self._size,
            length * self._low.tolist(),
            length * self._high.tolist(),
        )

    def encode_stack(self, val: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(val).astype(self._dtype)

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
    def get_encode_list(self):
        if self.is_discrete():
            arr = [
                RLBaseTypes.NP_ARRAY,
                RLBaseTypes.ARRAY_DISCRETE,
                RLBaseTypes.ARRAY_CONTINUOUS,
                RLBaseTypes.BOX,
            ]
        else:
            arr = [
                RLBaseTypes.NP_ARRAY,
                RLBaseTypes.ARRAY_CONTINUOUS,
                RLBaseTypes.BOX,
                RLBaseTypes.ARRAY_DISCRETE,
            ]
        arr += [
            RLBaseTypes.DISCRETE,
            RLBaseTypes.TEXT,
            RLBaseTypes.MULTI,
            # RLBaseTypes.CONTINUOUS, NG
        ]
        return arr

    # --- DiscreteSpace
    def create_encode_space_DiscreteSpace(self):
        from srl.base.spaces.discrete import DiscreteSpace

        assert self.division_tbl is not None, "Call 'create_division_tbl(division_num)' first"
        return DiscreteSpace(len(self.division_tbl))  # startは0

    def encode_to_space_DiscreteSpace(self, val: np.ndarray) -> int:
        assert self.division_tbl is not None, "Call 'create_division_tbl(division_num)' first"
        d = np.sum(np.abs(self.division_tbl - val), axis=1)
        return int(np.argmin(d))

    def decode_from_space_DiscreteSpace(self, val: int) -> np.ndarray:
        if self.division_tbl is None:
            return np.full((self._size,), val, dtype=self._dtype)
        else:
            return self.division_tbl[val]

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

    def encode_to_space_ArrayDiscreteSpace(self, val: np.ndarray) -> List[int]:
        if self.division_tbl is None:
            return [int(round(v)) for v in val]
        else:
            return [self.encode_to_space_DiscreteSpace(val)]

    def decode_from_space_ArrayDiscreteSpace(self, val: List[int]) -> np.ndarray:
        if self.division_tbl is None:
            return np.asarray(val, dtype=self._dtype)
        else:
            return self.division_tbl[val[0]]

    # --- ContinuousSpace
    def create_encode_space_ContinuousSpace(self):
        raise NotSupportedError()

    def encode_to_space_ContinuousSpace(self, val: np.ndarray) -> float:
        raise NotSupportedError()

    def decode_from_space_ContinuousSpace(self, val: float) -> np.ndarray:
        raise NotSupportedError()

    # --- ArrayContinuousSpace
    def create_encode_space_ArrayContinuousSpace(self):
        from srl.base.spaces.array_continuous import ArrayContinuousSpace

        return ArrayContinuousSpace(self._size, self._low, self._high)

    def encode_to_space_ArrayContinuousSpace(self, val: np.ndarray) -> List[float]:
        return val.tolist()

    def decode_from_space_ArrayContinuousSpace(self, val: List[float]) -> np.ndarray:
        return np.asarray(val, dtype=self._dtype)

    # --- NpArray
    def create_encode_space_NpArraySpace(self, options: SpaceEncodeOptions):
        dtype = options.cast_dtype if options.cast else self._dtype
        return self.copy(dtype=dtype)

    def encode_to_space_NpArraySpace(self, x: np.ndarray, to_space: SpaceBase) -> np.ndarray:
        if to_space.encode_options.cast:
            x = x.astype(to_space.dtype)
        return x

    def decode_from_space_NpArraySpace(self, x: np.ndarray, from_space: SpaceBase) -> np.ndarray:
        if from_space.encode_options.cast:
            x = x.astype(self._dtype)
        return x

    # --- Box
    def create_encode_space_Box(self, options: SpaceEncodeOptions):
        from srl.base.spaces.box import BoxSpace

        dtype = options.cast_dtype if options.cast else self._dtype
        return BoxSpace((self._size,), self._low, self._high, dtype, self.stype)

    def encode_to_space_Box(self, x: np.ndarray, to_space: SpaceBase) -> np.ndarray:
        if to_space.encode_options.cast:
            x = x.astype(to_space.dtype)
        return x

    def decode_from_space_Box(self, x: np.ndarray, from_space: SpaceBase) -> np.ndarray:
        if from_space.encode_options.cast:
            x = x.astype(self._dtype)
        return x

    # --- TextSpace
    def create_encode_space_TextSpace(self):
        from srl.base.spaces.text import TextSpace

        return TextSpace(min_length=1, charset="0123456789-.,")

    def encode_to_space_TextSpace(self, val: np.ndarray) -> str:
        return ",".join([str(v) for v in val.tolist()])

    def decode_from_space_TextSpace(self, val: str) -> np.ndarray:
        return np.array([float(v) for v in val.split(",")], dtype=self._dtype)

    # --- Multi
    def create_encode_space_MultiSpace(self):
        from srl.base.spaces.multi import MultiSpace

        return MultiSpace([self.copy()])

    def encode_to_space_MultiSpace(self, val: np.ndarray) -> list:
        return [val]

    def decode_from_space_MultiSpace(self, val: list) -> np.ndarray:
        return val[0]
