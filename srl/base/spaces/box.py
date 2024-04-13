import logging
import random
import time
from typing import Any, List, Tuple, Union

import numpy as np

from srl.base.define import SpaceTypes
from srl.base.exception import NotSupportedError

from .space import SpaceBase

logger = logging.getLogger(__name__)


class BoxSpace(SpaceBase[np.ndarray]):
    def __init__(
        self,
        shape: Union[List[int], Tuple[int, ...]],
        low: Union[float, List[float], Tuple[float, ...], np.ndarray] = -np.inf,
        high: Union[float, List[float], Tuple[float, ...], np.ndarray] = np.inf,
        dtype: Any = np.float32,
        stype: SpaceTypes = SpaceTypes.UNKNOWN,
    ) -> None:
        self._low: np.ndarray = np.full(shape, low, dtype=dtype) if np.isscalar(low) else np.asarray(low)
        self._high: np.ndarray = np.full(shape, high, dtype=dtype) if np.isscalar(high) else np.asarray(high)
        self._shape = tuple(shape)
        self._dtype = dtype
        if stype == SpaceTypes.UNKNOWN:
            self._stype = SpaceTypes.DISCRETE if "int" in str(dtype) else SpaceTypes.CONTINUOUS
        else:
            self._stype = stype

        assert self.shape == self.high.shape
        assert self.low.shape == self.high.shape
        assert np.less_equal(self.low, self.high).all()

        self._is_inf = np.isinf(low).any() or np.isinf(high).any()
        self.division_tbl = None
        self.decode_int_tbl = None
        self.encode_int_tbl = None
        self._f_stack_gray_2ch_shape = None

    @property
    def shape(self):
        return self._shape

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def stype(self) -> SpaceTypes:
        return self._stype

    @property
    def dtype(self):
        return self._dtype

    def sample(self, mask: List[np.ndarray] = []) -> np.ndarray:
        if self._stype == SpaceTypes.DISCRETE:
            if len(mask) > 0:
                self._create_int_tbl()
                assert self.encode_int_tbl is not None
                mask2 = [tuple(m.flatten().tolist()) for m in mask]
                valid_acts = [k for k in self.encode_int_tbl.keys() if k not in mask2]
                a = random.choice(valid_acts)
                return np.array(a, self._dtype).reshape(self._shape)
            else:
                return np.random.randint(self._low, self._high + 1, dtype=self._dtype)
        else:
            if len(mask) > 0:
                logger.info(f"mask is not support: {mask}")
            if self._is_inf:
                # infの場合は正規分布に従う乱数
                return np.random.normal(size=self.shape)
            r = np.random.random_sample(self.shape)
            return self.low + r * (self.high - self.low)

    def get_valid_actions(self, masks: List[np.ndarray] = []) -> List[np.ndarray]:
        if self._stype == SpaceTypes.DISCRETE:
            self._create_int_tbl()
            assert self.decode_int_tbl is not None
            acts = [np.array(v, self._dtype).reshape(self._shape) for v in self.decode_int_tbl]
            valid_acts = []
            for a in acts:
                f = True
                for m in masks:
                    if (a == m).all():
                        f = False
                        break
                if f:
                    valid_acts.append(a)
            return valid_acts
        else:
            raise NotSupportedError()

    def sanitize(self, val: Any) -> np.ndarray:
        val = np.asarray(val, self._dtype).reshape(self._shape)
        return np.clip(val, self._low, self._high)

    def check_val(self, val: Any) -> bool:
        if not isinstance(val, np.ndarray):
            return False
        if self._shape != val.shape:
            return False
        if (val < self._low).any():
            return False
        if (val > self._high).any():
            return False
        return True

    def to_str(self, val: np.ndarray) -> str:
        return str(val.flatten().tolist()).replace(" ", "")[1:-1]

    def get_default(self) -> np.ndarray:
        return np.zeros(self.shape)

    def copy(self) -> "BoxSpace":
        o = BoxSpace(self._shape, self._low, self._high, self._dtype, self._stype)
        o.division_tbl = self.division_tbl
        o.decode_int_tbl = self.decode_int_tbl
        o.encode_int_tbl = self.encode_int_tbl
        o._f_stack_gray_2ch_shape = self._f_stack_gray_2ch_shape
        return o

    def __eq__(self, o: "BoxSpace") -> bool:
        if not isinstance(o, BoxSpace):
            return False
        return (
            (self._shape == o._shape)
            and (self._low == o._low).all()
            and (self._high == o._high).all()
            and (self._dtype == o._dtype)
            and (self._stype == o._stype)
        )

    def __str__(self) -> str:
        if self.division_tbl is None:
            s = ""
        else:
            s = f", division({self.int_size})"
        return f"Box{self.shape}, range[{np.min(self.low)}, {np.max(self.high)}]{s}, {self._dtype}, {self._stype}"

    # --- stack
    def create_stack_space(self, length: int):
        if self._stype == SpaceTypes.GRAY_2ch:
            self._f_stack_gray_2ch_shape = self._shape + (length,)
            return BoxSpace(
                self._shape + (length,),
                np.min(self._low),
                np.max(self._high),
                self._dtype,
                SpaceTypes.IMAGE,
            )
        else:
            return BoxSpace(
                (length,) + self._shape,
                np.min(self._low),
                np.max(self._high),
                self._dtype,
                self._stype,
            )

    def encode_stack(self, val: List[np.ndarray]) -> np.ndarray:
        if self._f_stack_gray_2ch_shape is not None:
            return np.asarray(val, self._dtype).reshape(self._f_stack_gray_2ch_shape)
        else:
            return np.asarray(val, self._dtype)

    # --------------------------------------
    # create_division_tbl
    # --------------------------------------
    def create_division_tbl(self, division_num: int) -> None:
        if self._stype == SpaceTypes.DISCRETE:
            return
        if self._is_inf:  # infは定義できない
            return
        if division_num <= 0:
            return

        import itertools

        low_flatten = self.low.flatten()
        high_flatten = self.high.flatten()

        N = len(low_flatten) ** division_num
        if N > 100_000:
            logger.warning(f"It may take some time.(N={N}={len(low_flatten)}**{division_num})")

        t0 = time.time()
        act_list = []
        for i in range(len(low_flatten)):
            low = low_flatten[i]
            high = high_flatten[i]
            diff = (high - low) / (division_num - 1)
            act_list.append([low + diff * j for j in range(division_num)])

        act_list = list(itertools.product(*act_list))
        self.division_tbl = np.reshape(act_list, (-1,) + self.shape).astype(self._dtype)
        n = len(self.division_tbl)

        logger.info(f"created division: {division_num}(n={n})({time.time() - t0:.3f}s)")

    def _create_int_tbl(self) -> None:
        if self.decode_int_tbl is not None:
            return
        import itertools

        # flattenしたのをkeyにする
        t0 = time.time()
        low = self._low.flatten().astype(np.int64)
        high = self._high.flatten().astype(np.int64)
        arr_list = [[a for a in range(low[i], high[i] + 1)] for i in range(len(low))]
        self.decode_int_tbl = list(itertools.product(*arr_list))
        self.encode_int_tbl = {}
        for i, v in enumerate(self.decode_int_tbl):
            self.encode_int_tbl[v] = i
        logger.info(f"create table time: {time.time() - t0:.1f}s")

    # --------------------------------------
    # action discrete
    # --------------------------------------
    @property
    def int_size(self) -> int:
        if self._stype == SpaceTypes.DISCRETE:
            self._create_int_tbl()
            assert self.decode_int_tbl is not None
            return len(self.decode_int_tbl)
        else:
            assert self.division_tbl is not None, "Call 'create_division_tbl(division_num)' first"
            return len(self.division_tbl)

    def encode_to_int(self, val: np.ndarray) -> int:
        if self._stype == SpaceTypes.DISCRETE:
            self._create_int_tbl()
            assert self.encode_int_tbl is not None
            key = val.flatten().tolist()
            return self.encode_int_tbl[tuple(key)]
        else:
            assert self.division_tbl is not None, "Call 'create_division_tbl(division_num)' first"
            # ユークリッド距離で一番近いものを選択
            d = (self.division_tbl - val).reshape((self.division_tbl.shape[0], -1))
            d = np.linalg.norm(d, axis=1)
            return int(np.argmin(d))

    def decode_from_int(self, val: int) -> np.ndarray:
        if self._stype == SpaceTypes.DISCRETE:
            self._create_int_tbl()
            assert self.decode_int_tbl is not None
            return np.array(self.decode_int_tbl[val], self._dtype).reshape(self._shape)
        else:
            if self.division_tbl is None:
                return np.full(self.shape, val, dtype=self._dtype)
            else:
                return self.division_tbl[val]

    # --------------------------------------
    # observation discrete
    # --------------------------------------
    @property
    def list_int_size(self) -> int:
        if self._stype == SpaceTypes.DISCRETE:
            return len(self.low.flatten())
        else:
            if self.division_tbl is None:
                return len(self.low.flatten())
            else:
                return 1

    @property
    def list_int_low(self) -> List[int]:
        if self._stype == SpaceTypes.DISCRETE:
            return self.low.flatten().tolist()
        else:
            if self.division_tbl is None:
                return np.round(self.low.flatten()).tolist()
            else:
                return [0]

    @property
    def list_int_high(self) -> List[int]:
        if self._stype == SpaceTypes.DISCRETE:
            return self.high.flatten().tolist()
        else:
            if self.division_tbl is None:
                return np.round(self.high.flatten()).tolist()
            else:
                return [self.int_size]

    def encode_to_list_int(self, val: np.ndarray) -> List[int]:
        if self._stype == SpaceTypes.DISCRETE:
            return [int(s) for s in val.flatten().tolist()]
        else:
            if self.division_tbl is None:
                # 分割してない場合は、roundで丸めるだけ
                return [int(s) for s in np.round(val).flatten().tolist()]
            else:
                # 分割してある場合
                n = self.encode_to_int(val)
                return [n]

    def decode_from_list_int(self, val: List[int]) -> np.ndarray:
        if self._stype == SpaceTypes.DISCRETE:
            return np.array(val, self._dtype).reshape(self._shape)
        else:
            if self.division_tbl is None:
                return np.array(val, dtype=self._dtype).reshape(self.shape)
            else:
                return self.division_tbl[val[0]]

    # --------------------------------------
    # action continuous
    # --------------------------------------
    @property
    def list_float_size(self) -> int:
        return len(self.low.flatten())

    @property
    def list_float_low(self) -> List[float]:
        return self.low.flatten().tolist()

    @property
    def list_float_high(self) -> List[float]:
        return self.high.flatten().tolist()

    def encode_to_list_float(self, val: np.ndarray) -> List[float]:
        return [float(v) for v in val.flatten().tolist()]

    def decode_from_list_float(self, val: List[float]) -> np.ndarray:
        return np.array(val, dtype=self._dtype).reshape(self.shape)

    # --------------------------------------
    # observation continuous, image
    # --------------------------------------
    @property
    def np_shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def np_low(self) -> np.ndarray:
        return self._low

    @property
    def np_high(self) -> np.ndarray:
        return self._high

    def encode_to_np(self, val: np.ndarray, dtype) -> np.ndarray:
        val = val.astype(dtype)
        if val.shape == ():
            val = val.reshape((1,))
        return val

    def decode_from_np(self, val: np.ndarray) -> np.ndarray:
        return val.astype(self._dtype)
