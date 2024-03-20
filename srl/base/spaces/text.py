import logging
import random
from typing import Any, List, Tuple

import numpy as np

from srl.base.define import SpaceTypes

from .space import SpaceBase

alphanumeric = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


logger = logging.getLogger(__name__)


class TextSpace(SpaceBase[str]):
    def __init__(
        self,
        max_length: int,
        min_length: int = 0,
        sample_charset: str = alphanumeric,
    ) -> None:
        self._max_length = max_length
        self._min_length = min_length
        self._sample_charset = sample_charset

        assert 0 <= min_length
        assert min_length <= max_length

    @property
    def base_stype(self) -> SpaceTypes:
        return SpaceTypes.DISCRETE

    @property
    def dtype(self):
        return np.uint64

    def sample(self, mask: str = "") -> str:
        charset = [c for c in self._sample_charset if c not in mask]
        n = random.randint(self._min_length, self._max_length)
        text = [random.choice(charset) for _ in range(n)]
        return "".join(text)

    def sanitize(self, val: Any) -> str:
        if not isinstance(val, str):
            val = str(val)
        if len(val) < self._min_length:
            val += "".join([" " for _ in range(self._min_length - len(val))])
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

    def get_default(self) -> str:
        if self._min_length == 0:
            return ""
        return "".join([" " for _ in range(self._min_length)])

    def copy(self) -> "TextSpace":
        return TextSpace(self._min_length, self._max_length, self._sample_charset)

    def __eq__(self, o: "TextSpace") -> bool:
        return (
            self._min_length == o._min_length
            and self._max_length == o._max_length
            and self._sample_charset == o._sample_charset
        )

    def __str__(self) -> str:
        return f"Text({self._min_length}, {self._max_length})"

    # --------------------------------------
    # action discrete
    # --------------------------------------
    @property
    def int_size(self) -> int:
        self._create_tbl()
        assert self.decode_tbl is not None
        return len(self.decode_tbl)

    def encode_to_int(self, val: str) -> int:
        self._create_tbl()
        return self.encode_tbl[tuple(val)]

    def decode_from_int(self, val: int) -> str:
        self._create_tbl()
        assert self.decode_tbl is not None
        return list(self.decode_tbl[val])

    # --------------------------------------
    # observation discrete
    # --------------------------------------
    @property
    def list_int_size(self) -> int:
        """continuous list length"""
        raise NotImplementedError()

    @property
    def list_int_low(self) -> List[int]:
        """continuous list length range"""
        raise NotImplementedError()

    @property
    def list_int_high(self) -> List[int]:
        """continuous list length range"""
        raise NotImplementedError()

    def encode_to_list_int(self, val: str) -> List[int]:
        # 長さ揃えTODO
        return [ord(c) for c in val]

    def decode_from_list_int(self, val: List[int]) -> str:
        return "".join([chr(n) for n in val])

    # --------------------------------------
    # action continuous
    # --------------------------------------
    @property
    def list_float_size(self) -> int:
        return TODO

    @property
    def list_float_low(self) -> List[float]:
        return TODO

    @property
    def list_float_high(self) -> List[float]:
        return TODO

    def encode_to_list_float(self, val: str) -> List[float]:
        # 長さ揃えTODO
        return [ord(c) for c in val]

    def decode_from_list_float(self, val: List[float]) -> str:
        return "".join([chr(int(n)) for n in val])

    # --------------------------------------
    # observation continuous, image
    # --------------------------------------
    @property
    def np_shape(self) -> Tuple[int, ...]:
        return (self.list_size,)

    @property
    def np_low(self) -> np.ndarray:
        return np.array(self.list_low)

    @property
    def np_high(self) -> np.ndarray:
        return np.array(self.list_high)

    def encode_to_np(self, val: str, dtype) -> np.ndarray:
        return np.array(self.encode_to_list_float(val), dtype=dtype)

    def decode_from_np(self, val: np.ndarray) -> str:
        return self.decode_from_list_float(val.tolist())
