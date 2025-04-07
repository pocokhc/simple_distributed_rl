import logging
import random
from typing import Any, List, Tuple

import numpy as np

from srl.base.define import SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.space import SpaceBase

alphanumeric = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


logger = logging.getLogger(__name__)


class TextSpace(SpaceBase[str]):
    def __init__(
        self,
        max_length: int,
        min_length: int = 0,
        sample_charset: str = alphanumeric,
        padding: str = " ",
    ) -> None:
        self._max_length = max_length
        self._min_length = min_length
        self._sample_charset = sample_charset
        self._padding = padding

        assert len(self._sample_charset) > 0
        assert 0 <= min_length
        assert min_length <= max_length

    @property
    def stype(self) -> SpaceTypes:
        return SpaceTypes.DISCRETE

    @property
    def dtype(self):
        return np.uint64

    def sample(self, mask: str = "") -> str:
        charset = [c for c in self._sample_charset if c not in mask]
        n = random.randint(self._min_length, self._max_length)
        text = [random.choice(charset) for _ in range(n)]
        return "".join(text)

    def get_valid_actions(self, masks: List[str] = []) -> List[str]:
        raise NotImplementedError("TODO")  # 組み合わせ爆発どうするか未定

    def sanitize(self, val: Any) -> str:
        if not isinstance(val, str):
            val = str(val)
        if len(val) < self._max_length:
            val += "".join([" " for _ in range(self._max_length - len(val))])
        if len(val) > self._max_length:
            val = val[: self._max_length]
        return val

    def check_val(self, val: Any) -> bool:
        if not isinstance(val, str):
            return False
        if len(val) < self._min_length:
            return False
        if len(val) > self._max_length:
            return False
        return True

    def to_str(self, val: str) -> str:
        return val

    def get_default(self) -> str:
        if self._max_length == 0:
            return ""
        return "".join([self._sample_charset[0] for _ in range(self._max_length)])

    def copy(self, **kwargs) -> "TextSpace":
        keys = ["max_length", "min_length", "sample_charset", "padding"]
        args = [kwargs.get(key, getattr(self, f"_{key}")) for key in keys]
        return TextSpace(*args)

    def copy_value(self, v: str) -> str:
        return v

    def __eq__(self, o: "TextSpace") -> bool:
        if not isinstance(o, TextSpace):
            return False
        return self._min_length == o._min_length and self._max_length == o._max_length and self._sample_charset == o._sample_charset

    def __str__(self) -> str:
        return f"Text({self._min_length}, {self._max_length})"

    # --- stack
    def create_stack_space(self, length: int):
        return TextSpace(
            self._max_length * length,
            self._min_length,
            self._sample_charset,
            self._padding,
        )

    def encode_stack(self, val: List[str]) -> str:
        return "".join(val)

    # --------------------------------------
    # action discrete
    # --------------------------------------
    @property
    def int_size(self) -> int:
        raise NotImplementedError()

    def encode_to_int(self, val: str) -> int:
        raise NotImplementedError()

    def decode_from_int(self, val: int) -> str:
        raise NotImplementedError()

    # --------------------------------------
    # observation discrete
    # --------------------------------------
    @property
    def list_int_size(self) -> int:
        return self._max_length

    @property
    def list_int_low(self) -> List[int]:
        return [0] * self._max_length

    @property
    def list_int_high(self) -> List[int]:
        return [0x7F] * self._max_length

    def encode_to_list_int(self, val: str) -> List[int]:
        val = val + self._padding * (self._max_length - len(val))
        return [ord(c) for c in val]

    def decode_from_list_int(self, val: List[int]) -> str:
        return "".join([chr(n) for n in val])

    # --------------------------------------
    # action continuous
    # --------------------------------------
    @property
    def list_float_size(self) -> int:
        return self._max_length

    @property
    def list_float_low(self) -> List[float]:
        return [0.0] * self._max_length

    @property
    def list_float_high(self) -> List[float]:
        return [float(0x7F)] * self._max_length

    def encode_to_list_float(self, val: str) -> List[float]:
        val = val + self._padding * (self._max_length - len(val))
        return [float(ord(c)) for c in val]

    def decode_from_list_float(self, val: List[float]) -> str:
        return "".join([chr(int(n)) for n in val])

    # --------------------------------------
    # observation continuous, image
    # --------------------------------------
    @property
    def np_shape(self) -> Tuple[int, ...]:
        return (self._max_length,)

    @property
    def np_low(self) -> np.ndarray:
        return np.array([0.0] * self._max_length)

    @property
    def np_high(self) -> np.ndarray:
        return np.array([0x7F] * self._max_length)

    def encode_to_np(self, val: str, dtype) -> np.ndarray:
        return np.array(self.encode_to_list_float(val), dtype=dtype)

    def decode_from_np(self, val: np.ndarray) -> str:
        return self.decode_from_list_float(val.tolist())

    # --------------------------------------
    # spaces
    # --------------------------------------
    def create_encode_space(self, space_name: str) -> SpaceBase:
        from srl.base.spaces.array_continuous import ArrayContinuousSpace
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
            return BoxSpace(self.np_shape, self.np_low, self.np_high)
        elif space_name == "BoxSpace_float":
            return BoxSpace(self.np_shape, self.np_low, self.np_high, np.float32)
        elif space_name == "TextSpace":
            raise NotSupportedError()
        raise NotImplementedError(space_name)

    def encode_to_space(self, val: str, space: SpaceBase) -> Any:
        from srl.base.spaces.array_continuous import ArrayContinuousSpace
        from srl.base.spaces.array_discrete import ArrayDiscreteSpace
        from srl.base.spaces.box import BoxSpace
        from srl.base.spaces.continuous import ContinuousSpace
        from srl.base.spaces.discrete import DiscreteSpace
        from srl.base.spaces.multi import MultiSpace

        if isinstance(space, DiscreteSpace):
            raise NotImplementedError()
        elif isinstance(space, ArrayDiscreteSpace):
            val = val + self._padding * (self._max_length - len(val))
            return [ord(c) for c in val]
        elif isinstance(space, ContinuousSpace):
            raise NotImplementedError()
        elif isinstance(space, ArrayContinuousSpace):
            val = val + self._padding * (self._max_length - len(val))
            return [float(ord(c)) for c in val]
        elif isinstance(space, BoxSpace):
            return np.array(self.encode_to_list_float(val), dtype=space.dtype)
        elif isinstance(space, TextSpace):
            return val
        elif isinstance(space, MultiSpace):
            return val
        raise NotImplementedError()

    def decode_from_space(self, val: Any, space: SpaceBase) -> str:
        from srl.base.spaces.array_continuous import ArrayContinuousSpace
        from srl.base.spaces.array_discrete import ArrayDiscreteSpace
        from srl.base.spaces.box import BoxSpace
        from srl.base.spaces.continuous import ContinuousSpace
        from srl.base.spaces.discrete import DiscreteSpace
        from srl.base.spaces.multi import MultiSpace

        if isinstance(space, DiscreteSpace):
            raise NotImplementedError()
        elif isinstance(space, ArrayDiscreteSpace):
            return "".join([chr(n) for n in val])
        elif isinstance(space, ContinuousSpace):
            raise NotImplementedError()
        elif isinstance(space, ArrayContinuousSpace):
            return "".join([chr(int(n)) for n in val])
        elif isinstance(space, BoxSpace):
            return self.decode_from_list_float(val.tolist())
        elif isinstance(space, TextSpace):
            return val
        elif isinstance(space, MultiSpace):
            return val
        raise NotImplementedError()
