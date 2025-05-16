import logging
import random
from typing import Any, List

import numpy as np

from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.space import SpaceBase

alphanumeric = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


logger = logging.getLogger(__name__)


class TextSpace(SpaceBase[str]):
    def __init__(
        self,
        max_length: int = -1,
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
        if max_length > 0:
            assert min_length <= max_length

    @property
    def max_length(self):
        return self._max_length

    @property
    def min_length(self):
        return self._min_length

    @property
    def sample_charset(self):
        return self._sample_charset

    @property
    def padding(self):
        return self._padding

    @property
    def stype(self) -> SpaceTypes:
        return SpaceTypes.DISCRETE

    @property
    def dtype(self):
        return np.uint64

    def sample(self, mask: str = "") -> str:
        if self._max_length <= 0:
            raise NotSupportedError()
        charset = [c for c in self._sample_charset if c not in mask]
        n = random.randint(self._min_length, self._max_length)
        text = [random.choice(charset) for _ in range(n)]
        return "".join(text)

    def get_valid_actions(self, masks: List[str] = []) -> List[str]:
        raise NotImplementedError("TODO")  # 組み合わせ爆発どうするか未定

    def sanitize(self, val: Any) -> str:
        if not isinstance(val, str):
            val = str(val)
        if self._max_length > 0:
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
        if self._max_length > 0:
            if len(val) > self._max_length:
                return False
        return True

    def to_str(self, val: str) -> str:
        return val

    def get_default(self) -> str:
        if self._max_length <= 0:
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
            self._max_length * length if self._max_length > 0 else -1,
            self._min_length,
            self._sample_charset,
            self._padding,
        )

    def encode_stack(self, val: List[str]) -> str:
        return "".join(val)

    # --------------------------------------
    # spaces
    # --------------------------------------
    def get_encode_type_list(self):
        priority_list = [RLBaseTypes.TEXT]
        exclude_list = [RLBaseTypes.DISCRETE, RLBaseTypes.CONTINUOUS]
        return priority_list, exclude_list

    # --- DiscreteSpace
    def create_encode_space_DiscreteSpace(self):
        raise NotSupportedError()

    def encode_to_space_DiscreteSpace(self, val: str) -> int:
        raise NotSupportedError()

    def decode_from_space_DiscreteSpace(self, val: int) -> str:
        raise NotSupportedError()

    # --- ArrayDiscreteSpace
    def create_encode_space_ArrayDiscreteSpace(self):
        if self._max_length <= 0:
            raise NotSupportedError()

        from srl.base.spaces.array_discrete import ArrayDiscreteSpace

        return ArrayDiscreteSpace(self._max_length, 0, 0x7F)

    def encode_to_space_ArrayDiscreteSpace(self, val: str) -> List[int]:
        val = val + self._padding * (self._max_length - len(val))
        return [ord(c) for c in val]

    def decode_from_space_ArrayDiscreteSpace(self, val: List[int]) -> str:
        return "".join([chr(n) for n in val])

    # --- ContinuousSpace
    def create_encode_space_ContinuousSpace(self):
        raise NotSupportedError()

    def encode_to_space_ContinuousSpace(self, val: str) -> float:
        raise NotSupportedError()

    def decode_from_space_ContinuousSpace(self, val: float) -> str:
        raise NotSupportedError()

    # --- ArrayContinuousSpace
    def create_encode_space_ArrayContinuousSpace(self):
        if self._max_length <= 0:
            raise NotSupportedError()

        from srl.base.spaces.array_continuous import ArrayContinuousSpace

        return ArrayContinuousSpace(self._max_length, 0.0, float(0x7F))

    def encode_to_space_ArrayContinuousSpace(self, val: str) -> List[float]:
        val = val + self._padding * (self._max_length - len(val))
        return [float(ord(c)) for c in val]

    def decode_from_space_ArrayContinuousSpace(self, val: List[float]) -> str:
        return "".join([chr(int(n)) for n in val])

    # --- Box
    def create_encode_space_Box(self, space_type: RLBaseTypes, np_dtype):
        from srl.base.spaces.box import BoxSpace

        # TODO: Box Image

        return BoxSpace((self._max_length,), 0, 0x7F, np_dtype, stype=SpaceTypes.CONTINUOUS)

    def encode_to_space_Box(self, val: str, space) -> np.ndarray:
        return np.array(self.encode_to_space_ArrayDiscreteSpace(val), dtype=space.dtype)

    def decode_from_space_Box(self, val: np.ndarray, space) -> str:
        return self.decode_from_space_ArrayDiscreteSpace(val.tolist())

    # --- TextSpace
    def create_encode_space_TextSpace(self):
        return self.copy()

    def encode_to_space_TextSpace(self, val: str) -> str:
        return val

    def decode_from_space_TextSpace(self, val: str) -> str:
        return val
