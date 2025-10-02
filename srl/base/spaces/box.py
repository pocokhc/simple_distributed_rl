import logging
import random
import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.space import SpaceBase, SpaceEncodeOptions

logger = logging.getLogger(__name__)


class BoxSpace(SpaceBase[np.ndarray]):
    def __init__(
        self,
        shape: Union[List[int], Tuple[int, ...]],
        low: Union[float, List[float], Tuple[float, ...], np.ndarray] = -np.inf,
        high: Union[float, List[float], Tuple[float, ...], np.ndarray] = np.inf,
        dtype: Any = np.float32,
        stype: SpaceTypes = SpaceTypes.UNKNOWN,
        is_stack_ch: Optional[bool] = None,  # None is auto
    ) -> None:
        super().__init__()
        self._low: np.ndarray = np.full(shape, low) if np.isscalar(low) else np.asarray(low)
        self._high: np.ndarray = np.full(shape, high) if np.isscalar(high) else np.asarray(high)
        self._low = self._low.astype(dtype)
        self._high = self._high.astype(dtype)
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
        if is_stack_ch is None:
            if self._stype in [SpaceTypes.GRAY_HW, SpaceTypes.GRAY_HW1]:
                self._is_stack_ch = True
            else:
                self._is_stack_ch = False
        else:
            self._is_stack_ch = is_stack_ch

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
    def flatten_size(self) -> int:
        return int(np.prod(self._shape))

    def rescale_from(self, x: np.ndarray, src_low: float, src_high: float, raise_error: bool = False) -> np.ndarray:
        """
        値 x を [src_low, src_high] から [self._low, self._high] に線形変換する。
        self._low/self._high が無限大の場合は例外を出すか、そのまま返す。

        Parameters
        ----------
        x : np.ndarray
            入力値
        src_low : float
            入力値の下限
        src_high : float
            入力値の上限 (src_low < src_high でなければならない)
        raise_error : bool, optional
            True の場合は不正な範囲に対して例外を出す (default=False)

        Returns
        -------
        np.ndarray
            スケーリング後の値

        Raises
        ------
        AssertionError
            src_low >= src_high の場合
        ValueError
            self._low または self._high が inf かつ raise_error=True の場合
        """
        assert src_low < src_high, f"src_low >= src_high: {src_low}, {src_high}"
        if self._is_inf:
            if raise_error:
                raise ValueError(f"low/high is inf. {self}")
            else:
                return x
        scale = (self._high - self._low) / (src_high - src_low)
        return (x - src_low) * scale + self._low

    def rescale_to(self, x: np.ndarray, target_low: float, target_high: float, raise_error: bool = False) -> np.ndarray:
        """
        値 x を [self.low, self.high] から [target_low, target_high] に線形変換する。
        low/high が無限大の場合は例外を出すか、そのまま返す。

        Parameters
        ----------
        x : np.ndarray
            入力値
        target_low : float
            変換後の最小値
        target_high : float
            変換後の最大値
        raise_error : bool, optional
            True の場合は不正な範囲に対して例外を出す (default=False)

        Returns
        -------
        np.ndarray
            スケーリング後の値
        """
        if self._is_inf:
            if raise_error:
                raise ValueError(f"low/high is inf. {self}")
            return x
        if (self.low == self.high).any():
            if raise_error:
                raise ValueError(f"low == high is not allowed. {self}")
            return np.full_like(x, target_low)
        return (x - self.low) / (self.high - self.low) * (target_high - target_low) + target_low

    def to_image(self, x: np.ndarray) -> np.ndarray:
        """
        入力された x を画像データとして uint8 に変換する。
        """

        # 無限範囲の場合はスケーリング不可
        if self._is_inf:
            raise ValueError("Cannot convert to image when low or high is infinite.")

        if self._stype == SpaceTypes.GRAY_HW:
            assert x.ndim == 2, f"{self._stype.name} expects (H, W), got {x.shape}"
            x = x[..., None]  # (H, W, 1)
            x = np.repeat(x, 3, axis=2)  # (H, W, 3)
        elif self._stype == SpaceTypes.GRAY_HW1:
            assert x.ndim == 3 and x.shape[2] == 1, f"{self._stype.name} expects (H, W, 1), got {x.shape}"
            x = np.repeat(x, 3, axis=2)  # (H, W, 3)
        elif self._stype in {SpaceTypes.RGB, SpaceTypes.IMAGE_MAP, SpaceTypes.FEATURE_MAP}:
            assert x.ndim == 3, f"{self._stype.name} expects 3D shape (H, W, C), got {x.shape}"
            if x.shape[2] == 1:
                x = np.repeat(x, 3, axis=2)  # (H, W, 1) → (H, W, 3)
            elif x.shape[2] != 3:
                raise ValueError(f"{self._stype.name} expects channel=3, got {x.shape[2]}")
        else:
            raise ValueError(f"Unsupported stype for image conversion: {self._stype.name}")

        # スケーリング
        high = np.max(self.high)
        low = np.min(self.low)
        if low < high:
            scale = 255.0 / (high - low)
            offset = -low * scale
            x = x * scale + offset
        x_clipped = np.clip(x, 0, 255)

        return x_clipped.astype(np.uint8)

    # ----------------------------------------

    @property
    def stype(self) -> SpaceTypes:
        return self._stype

    @property
    def dtype(self):
        return self._dtype

    @property
    def name(self) -> str:
        return "Box"

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
                return np.random.randint(self._low, self._high, dtype=self._dtype)
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
        return np.asarray(val, self._dtype).reshape(self._shape).clip(self._low, self._high)

    def check_val(self, val: Any) -> bool:
        if not isinstance(val, np.ndarray):
            return False
        if self._shape != val.shape:
            return False
        if np.isnan(val).any():
            return False
        if (val < self._low).any():
            return False
        if (val > self._high).any():
            return False
        return True

    def to_str(self, val: np.ndarray) -> str:
        return str(val.flatten().tolist()).replace(" ", "")[1:-1]

    def get_default(self) -> np.ndarray:
        return np.zeros(self.shape, self._dtype)

    def copy(self, **kwargs) -> "BoxSpace":
        keys = ["shape", "low", "high", "dtype", "stype", "is_stack_ch"]
        args = [kwargs.get(key, getattr(self, f"_{key}")) for key in keys]
        o = BoxSpace(*args)
        o.division_tbl = self.division_tbl
        o.decode_int_tbl = self.decode_int_tbl
        o.encode_int_tbl = self.encode_int_tbl
        return o

    def copy_value(self, v: np.ndarray) -> np.ndarray:
        return v.copy()

    def equal_val(self, v1: np.ndarray, v2: np.ndarray, rtol=1e-9, atol=1e-12) -> bool:
        if v1.shape != v2.shape:
            return False
        if np.issubdtype(v1.dtype, np.floating) or np.issubdtype(v2.dtype, np.floating):
            return np.allclose(v1, v2, rtol=rtol, atol=atol)
        return np.array_equal(v1, v2)

    def __eq__(self, o: "BoxSpace") -> bool:
        if not isinstance(o, BoxSpace):
            return False
        return (
            (self._shape == o._shape)
            and (self._low == o._low).all()
            and (self._high == o._high).all()
            and (self._dtype == o._dtype)  #
            and (self._stype == o._stype)
        )

    def __str__(self) -> str:
        if self.division_tbl is None:
            s = ""
        else:
            s = f", division({len(self.division_tbl)})"
        return f"Box{self.shape}, range[{np.min(self.low)}, {np.max(self.high)}]{s}, {self._dtype}, {self._stype.name}"

    # --- stack
    def create_stack_space(self, length: int):
        if self._is_stack_ch:
            return BoxSpace(
                (self._shape[0], self._shape[1], length),
                np.min(self._low),
                np.max(self._high),
                self._dtype,
                SpaceTypes.IMAGE_MAP,
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
        state = np.asarray(val, self._dtype)
        if self._is_stack_ch:
            if self._stype == SpaceTypes.GRAY_HW:
                state = np.transpose(state, (1, 2, 0))
            elif self._stype == SpaceTypes.GRAY_HW1:
                state = np.transpose(np.squeeze(state, axis=-1), (1, 2, 0))
            else:
                raise ValueError(self._stype)
        return state

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
        if self.is_discrete():
            return
        if self.is_image_like():
            return
        if self._is_inf:  # infは定義できない
            return
        if division_num <= 0:
            return

        low_flatten = self.low.flatten()
        high_flatten = self.high.flatten()

        # 各要素は分割後のデカルト積のサイズが division_num になるように分割
        # ただし、各要素の最低は2
        # division_num = division_num_one ** size
        division_num_one = round(division_num ** (1 / len(low_flatten)))
        if division_num_one < 2:
            division_num_one = 2

        import itertools

        t0 = time.time()
        div_list = []
        for i in range(len(low_flatten)):
            low = low_flatten[i]
            high = high_flatten[i]
            diff = (high - low) / (division_num_one - 1)
            div_list.append([low + diff * j for j in range(division_num_one)])

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
        self.division_tbl = np.reshape(div_prods, (-1,) + self.shape).astype(self._dtype)
        n = len(self.division_tbl)

        logger.info(f"created division: size={n}, create time={time.time() - t0:.3f}s")

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
    # spaces
    # --------------------------------------
    def get_encode_list(self):
        if self.stype == SpaceTypes.DISCRETE:
            arr = [
                RLBaseTypes.BOX,
                RLBaseTypes.NP_ARRAY,
                RLBaseTypes.ARRAY_DISCRETE,
                RLBaseTypes.ARRAY_CONTINUOUS,
            ]
        else:
            arr = [
                RLBaseTypes.BOX,
                RLBaseTypes.NP_ARRAY,
                RLBaseTypes.ARRAY_CONTINUOUS,
                RLBaseTypes.ARRAY_DISCRETE,
            ]
        arr += [
            RLBaseTypes.TEXT,
            RLBaseTypes.DISCRETE,
            RLBaseTypes.MULTI,
            # RLBaseTypes.CONTINUOUS, NG
        ]
        return arr

    # --- DiscreteSpace
    def _get_int_size(self) -> int:
        if self._stype == SpaceTypes.DISCRETE:
            self._create_int_tbl()
            assert self.decode_int_tbl is not None
            return len(self.decode_int_tbl)
        else:
            assert self.division_tbl is not None, "Call 'create_division_tbl(division_num)' first"
            return len(self.division_tbl)

    def _encode_to_int(self, val: np.ndarray) -> int:
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

    def _set_encode_space_DiscreteSpace(self, options: SpaceEncodeOptions):
        from srl.base.spaces.discrete import DiscreteSpace

        return DiscreteSpace(self._get_int_size())  # startは0

    def _encode_to_space_DiscreteSpace(self, val: np.ndarray, **kwargs) -> int:
        return self._encode_to_int(val)

    def _decode_from_space_DiscreteSpace(self, val: int, **kwargs) -> np.ndarray:
        if self._stype == SpaceTypes.DISCRETE:
            self._create_int_tbl()
            assert self.decode_int_tbl is not None
            return np.array(self.decode_int_tbl[val], self._dtype).reshape(self._shape)
        else:
            if self.division_tbl is None:
                return np.full(self._shape, val, dtype=self._dtype)
            else:
                return self.division_tbl[val]

    # --- ArrayDiscreteSpace
    def _set_encode_space_ArrayDiscreteSpace(self, options: SpaceEncodeOptions):
        from srl.base.spaces.array_discrete import ArrayDiscreteSpace

        if self._stype == SpaceTypes.DISCRETE:
            return ArrayDiscreteSpace(
                len(self._low.flatten()),
                self._low.flatten().tolist(),
                self._high.flatten().tolist(),
            )

        if self.division_tbl is None:
            return ArrayDiscreteSpace(
                len(self._low.flatten()),
                np.round(self._low.flatten()).tolist(),
                np.round(self._high.flatten()).tolist(),
            )
        else:
            return ArrayDiscreteSpace(
                1,
                [0],
                [self._get_int_size()],
            )

    def _encode_to_space_ArrayDiscreteSpace(self, val: np.ndarray) -> List[int]:
        if self._stype == SpaceTypes.DISCRETE:
            return [int(s) for s in val.flatten().tolist()]
        else:
            if self.division_tbl is None:
                # 分割してない場合は、roundで丸めるだけ
                return [int(s) for s in np.round(val).flatten().tolist()]
            else:
                # 分割してある場合
                return [self._encode_to_int(val)]

    def _decode_from_space_ArrayDiscreteSpace(self, val: List[int]) -> np.ndarray:
        if self._stype == SpaceTypes.DISCRETE:
            return np.array(val, self._dtype).reshape(self._shape)
        else:
            if self.division_tbl is None:
                return np.array(val, dtype=self._dtype).reshape(self._shape)
            else:
                return self.division_tbl[val[0]]

    # --- ContinuousSpace
    def _set_encode_space_ContinuousSpace(self, options: SpaceEncodeOptions):
        raise NotSupportedError()

    def _encode_to_space_ContinuousSpace(self, val: np.ndarray) -> float:
        raise NotSupportedError()

    def _decode_from_space_ContinuousSpace(self, val: float) -> np.ndarray:
        raise NotSupportedError()

    # --- ArrayContinuousSpace
    def _set_encode_space_ArrayContinuousSpace(self, options: SpaceEncodeOptions):
        from srl.base.spaces.array_continuous import ArrayContinuousSpace

        return ArrayContinuousSpace(
            len(self._low.flatten()),
            self._low.flatten(),
            self._high.flatten(),
        )

    def _encode_to_space_ArrayContinuousSpace(self, val: np.ndarray) -> List[float]:
        return [float(v) for v in val.flatten().tolist()]

    def _decode_from_space_ArrayContinuousSpace(self, val: List[float]) -> np.ndarray:
        return np.array(val, dtype=self._dtype).reshape(self._shape)

    # --- NpArray
    def _set_encode_space_NpArraySpace(self, options: SpaceEncodeOptions):
        from srl.base.spaces.np_array import NpArraySpace

        if options.np_norm_type == "0to1":
            dtype = options.cast_dtype if options.cast else np.float32
            return NpArraySpace(len(self._low.flatten()), 0, 1, dtype, SpaceTypes.CONTINUOUS)
        elif options.np_norm_type == "-1to1":
            dtype = options.cast_dtype if options.cast else np.float32
            return NpArraySpace(len(self._low.flatten()), -1, 1, dtype, SpaceTypes.CONTINUOUS)

        if options.np_zero_start:
            low = 0
            high = self._high.flatten() - self._low.flatten()
        else:
            low = self._low.flatten()
            high = self._high.flatten()

        dtype = options.cast_dtype if options.cast else self._dtype
        return NpArraySpace(len(self._low.flatten()), low, high, dtype, self._stype)

    def _encode_to_space_NpArraySpace(self, x: np.ndarray) -> np.ndarray:
        if self.encode_options.np_norm_type == "0to1":
            x = (x - self._low) / (self._high - self._low)
        elif self.encode_options.np_norm_type == "-1to1":
            x = 2 * (x - self._low) / (self._high - self._low) - 1
        elif self.encode_options.np_zero_start:
            x = x - self._low

        x = x.flatten()
        if self.encode_options.cast:
            x = x.astype(self.encode_space.dtype)
        return x

    def _decode_from_space_NpArraySpace(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(self._shape)
        if self.encode_options.cast:
            x = x.astype(self._dtype)

        if self.encode_options.np_norm_type == "0to1":
            x = x * (self._high - self._low) + self._low
        elif self.encode_options.np_norm_type == "-1to1":
            x = ((x + 1) / 2) * (self._high - self._low) + self._low
        elif self.encode_options.np_zero_start:
            x = x + self._low
        return x

    # --- Box
    def _set_encode_space_Box(self, options: SpaceEncodeOptions):
        if options.np_norm_type == "0to1":
            dtype = options.cast_dtype if options.cast else np.float32
            return BoxSpace(self._shape, 0, 1, dtype, SpaceTypes.CONTINUOUS)
        elif options.np_norm_type == "-1to1":
            dtype = options.cast_dtype if options.cast else np.float32
            return BoxSpace(self._shape, -1, 1, dtype, SpaceTypes.CONTINUOUS)
        elif options.np_zero_start:
            dtype = options.cast_dtype if options.cast else self._dtype
            return BoxSpace(self._shape, 0, self._high - self._low, dtype, self._stype)

        return self.copy(dtype=options.cast_dtype if options.cast else self._dtype)

    def _encode_to_space_Box(self, x: np.ndarray) -> np.ndarray:
        if x.shape == ():
            x = x.reshape((1,))

        if self.encode_options.np_norm_type == "0to1":
            x = (x - self._low) / (self._high - self._low)
        elif self.encode_options.np_norm_type == "-1to1":
            x = 2 * (x - self._low) / (self._high - self._low) - 1
        elif self.encode_options.np_zero_start:
            x = x - self._low

        if self.encode_options.cast:
            x = x.astype(self.encode_space.dtype)
        return x

    def _decode_from_space_Box(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(self._shape)
        if self.encode_options.cast:
            x = x.astype(self._dtype)

        if self.encode_options.np_norm_type == "0to1":
            x = x * (self._high - self._low) + self._low
        elif self.encode_options.np_norm_type == "-1to1":
            x = ((x + 1) / 2) * (self._high - self._low) + self._low
        elif self.encode_options.np_zero_start:
            x = x + self._low
        return x

    # --- TextSpace
    def _set_encode_space_TextSpace(self, options: SpaceEncodeOptions):
        from srl.base.spaces.text import TextSpace

        return TextSpace(min_length=1, charset="0123456789-.,")

    def _encode_to_space_TextSpace(self, val: np.ndarray) -> str:
        return ",".join([str(v) for v in val.flatten().tolist()])

    def _decode_from_space_TextSpace(self, val: str) -> np.ndarray:
        return np.array([float(v) for v in val.split(",")], self._dtype).reshape(self._shape)

    # --- Multi
    def _set_encode_space_MultiSpace(self, options: SpaceEncodeOptions):
        from srl.base.spaces.multi import MultiSpace

        return MultiSpace([self.copy()])

    def _encode_to_space_MultiSpace(self, val: np.ndarray) -> list:
        return [val]

    def _decode_from_space_MultiSpace(self, val: list) -> np.ndarray:
        return val[0]
