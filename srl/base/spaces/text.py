import logging
import random
from typing import Any, List

import numpy as np

from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.space import SpaceBase, SpaceEncodeOptions

alphanumeric = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


logger = logging.getLogger(__name__)


class TextSpace(SpaceBase[str]):
    def __init__(
        self,
        max_length: int = -1,
        min_length: int = 0,
        charset: str = alphanumeric,
        padding: str = " ",
    ) -> None:
        super().__init__()
        self._max_length = max_length
        self._min_length = min_length
        self._charset = charset
        self._padding = padding

        assert len(self._charset) > 0
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
    def charset(self):
        return self._charset

    @property
    def padding(self):
        return self._padding

    # --------------------------------------

    @property
    def name(self) -> str:
        return "Text"

    @property
    def stype(self) -> SpaceTypes:
        return SpaceTypes.DISCRETE

    @property
    def dtype(self):
        return np.uint

    def sample(self, mask: str = "") -> str:
        if self._max_length <= 0:
            raise NotSupportedError()
        charset = [c for c in self._charset if c not in mask]
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
        return "".join([self._charset[0] for _ in range(self._max_length)])

    def copy(self, **kwargs) -> "TextSpace":
        keys = ["max_length", "min_length", "charset", "padding"]
        args = [kwargs.get(key, getattr(self, f"_{key}")) for key in keys]
        return TextSpace(*args)

    def copy_value(self, v: str) -> str:
        return v

    def equal_val(self, v1: str, v2: str) -> bool:
        return v1 == v2

    def __eq__(self, o: "TextSpace") -> bool:
        if not isinstance(o, TextSpace):
            return False
        return self._min_length == o._min_length and self._max_length == o._max_length and self._charset == o._charset

    def __str__(self) -> str:
        return f"Text({self._min_length}, {self._max_length})"

    # --- stack
    def create_stack_space(self, length: int):
        return TextSpace(
            self._max_length * length if self._max_length > 0 else -1,
            self._min_length,
            self._charset,
            self._padding,
        )

    def encode_stack(self, val: List[str]) -> str:
        return "".join(val)

    # --------------------------------------
    # spaces
    # --------------------------------------
    def get_encode_list(self):
        arr = [
            RLBaseTypes.TEXT,
            RLBaseTypes.ARRAY_DISCRETE,
            RLBaseTypes.ARRAY_CONTINUOUS,
            RLBaseTypes.NP_ARRAY,
            RLBaseTypes.BOX,
            RLBaseTypes.MULTI,
            # RLBaseTypes.DISCRETE, NG
            # RLBaseTypes.CONTINUOUS, NG
        ]
        return arr

    def get_encode_type_list(self):
        priority_list = [RLBaseTypes.TEXT]
        exclude_list = [RLBaseTypes.DISCRETE, RLBaseTypes.CONTINUOUS]
        return priority_list, exclude_list

    # --- DiscreteSpace
    def _set_encode_space_DiscreteSpace(self, options: SpaceEncodeOptions):
        from srl.base.spaces.discrete import DiscreteSpace

        if self._max_length <= 0:
            raise NotSupportedError()
        return DiscreteSpace(10**self._max_length - 1)

    def _encode_to_space_DiscreteSpace(self, val: str) -> int:
        return int(val)

    def _decode_from_space_DiscreteSpace(self, val: int) -> str:
        return str(val)

    # --- ArrayDiscreteSpace
    def _set_encode_space_ArrayDiscreteSpace(self, options: SpaceEncodeOptions):
        if self._max_length <= 0:
            raise NotSupportedError()

        from srl.base.spaces.array_discrete import ArrayDiscreteSpace

        return ArrayDiscreteSpace(self._max_length, 0, 0x7F)

    def _encode_to_space_ArrayDiscreteSpace(self, val: str) -> List[int]:
        val = val + self._padding * (self._max_length - len(val))
        return [ord(c) for c in val]

    def _decode_from_space_ArrayDiscreteSpace(self, val: List[int]) -> str:
        return "".join([chr(n) for n in val])

    # --- ContinuousSpace
    def _set_encode_space_ContinuousSpace(self, options: SpaceEncodeOptions):
        from srl.base.spaces.continuous import ContinuousSpace

        if self._max_length <= 0:
            raise NotSupportedError()
        return ContinuousSpace()

    def _encode_to_space_ContinuousSpace(self, val: str) -> float:
        return float(val)

    def _decode_from_space_ContinuousSpace(self, val: float) -> str:
        return str(val)

    # --- ArrayContinuousSpace
    def _set_encode_space_ArrayContinuousSpace(self, options: SpaceEncodeOptions):
        from srl.base.spaces.array_continuous import ArrayContinuousSpace

        if self._max_length <= 0:
            raise NotSupportedError()
        return ArrayContinuousSpace(self._max_length, 0.0, float(0x7F))

    def _encode_to_space_ArrayContinuousSpace(self, val: str) -> List[float]:
        val = val + self._padding * (self._max_length - len(val))
        return [float(ord(c)) for c in val]

    def _decode_from_space_ArrayContinuousSpace(self, val: List[float]) -> str:
        return "".join([chr(int(n)) for n in val])

    # --- NpArray
    def _set_encode_space_NpArraySpace(self, options: SpaceEncodeOptions):
        from srl.base.spaces.np_array import NpArraySpace

        if self._max_length <= 0:
            raise NotSupportedError()

        if options.np_norm_type == "0to1":
            dtype = options.cast_dtype if options.cast else np.float32
            return NpArraySpace(self._max_length, 0, 1, dtype, SpaceTypes.CONTINUOUS)
        elif options.np_norm_type == "-1to1":
            dtype = options.cast_dtype if options.cast else np.float32
            return NpArraySpace(self._max_length, -1, 1, dtype, SpaceTypes.CONTINUOUS)

        dtype = options.cast_dtype if options.cast else self.dtype
        return NpArraySpace(self._max_length, 0.0, float(0x7F), dtype, SpaceTypes.DISCRETE)

    def _encode_to_space_NpArraySpace(self, val: str) -> np.ndarray:
        arr_cont = self._encode_to_space_ArrayContinuousSpace(val)

        if self.encode_options.np_norm_type == "0to1":
            return np.array(
                [v / float(0x7F) for v in arr_cont],
                dtype=self.encode_space.dtype,
            )
        elif self.encode_options.np_norm_type == "-1to1":
            return np.array(
                [2 * v / float(0x7F) - 1 for v in arr_cont],
                dtype=self.encode_space.dtype,
            )

        return np.array(arr_cont, dtype=self.encode_space.dtype)

    def _decode_from_space_NpArraySpace(self, val: np.ndarray) -> str:
        if self.encode_options.np_norm_type == "0to1":
            val2 = [v * float(0x7F) for v in val.tolist()]
        elif self.encode_options.np_norm_type == "-1to1":
            val2 = [((v + 1) / 2) * float(0x7F) for v in val.tolist()]
        else:
            val2 = val.tolist()

        return self._decode_from_space_ArrayContinuousSpace(val2)

    # --- Box
    def _set_encode_space_Box(self, options: SpaceEncodeOptions):
        if self._max_length <= 0:
            raise NotSupportedError()

        from srl.base.spaces.box import BoxSpace

        if options.np_norm_type == "0to1":
            dtype = options.cast_dtype if options.cast else np.float32
            return BoxSpace((self._max_length,), 0, 1, dtype, SpaceTypes.CONTINUOUS)
        elif options.np_norm_type == "-1to1":
            dtype = options.cast_dtype if options.cast else np.float32
            return BoxSpace((self._max_length,), -1, 1, dtype, SpaceTypes.CONTINUOUS)

        dtype = options.cast_dtype if options.cast else self.dtype
        return BoxSpace((self._max_length,), 0, 0x7F, dtype, SpaceTypes.DISCRETE)

    def _encode_to_space_Box(self, val: str) -> np.ndarray:
        arr_cont = self._encode_to_space_ArrayContinuousSpace(val)

        if self.encode_options.np_norm_type == "0to1":
            return np.array(
                [v / float(0x7F) for v in arr_cont],
                dtype=self.encode_space.dtype,
            )
        elif self.encode_options.np_norm_type == "-1to1":
            return np.array(
                [2 * v / float(0x7F) - 1 for v in arr_cont],
                dtype=self.encode_space.dtype,
            )

        return np.array(arr_cont, dtype=self.encode_space.dtype)

    def _decode_from_space_Box(self, val: np.ndarray) -> str:
        if self.encode_options.np_norm_type == "0to1":
            val2 = [v * float(0x7F) for v in val.tolist()]
        elif self.encode_options.np_norm_type == "-1to1":
            val2 = [((v + 1) / 2) * float(0x7F) for v in val.tolist()]
        else:
            val2 = val.tolist()

        return self._decode_from_space_ArrayContinuousSpace(val2)

    # --- TextSpace
    def _set_encode_space_TextSpace(self, options: SpaceEncodeOptions):
        return self.copy()

    def _encode_to_space_TextSpace(self, val: str) -> str:
        return val

    def _decode_from_space_TextSpace(self, val: str) -> str:
        return val

    # --- Multi
    def _set_encode_space_MultiSpace(self, options: SpaceEncodeOptions):
        from srl.base.spaces.multi import MultiSpace

        return MultiSpace([self.copy()])

    def _encode_to_space_MultiSpace(self, val: str) -> list:
        return [val]

    def _decode_from_space_MultiSpace(self, val: list) -> str:
        return val[0]
