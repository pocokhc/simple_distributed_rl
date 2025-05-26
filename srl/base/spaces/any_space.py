from typing import Any, List

import numpy as np

from srl.base.define import SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.space import SpaceBase


class AnySpace(SpaceBase[Any]):
    @property
    def name(self) -> str:
        return "Any"

    def sample(self, mask: List[Any] = []) -> Any:
        raise NotSupportedError()

    def sanitize(self, val: Any) -> Any:
        return val

    def check_val(self, val: Any) -> bool:
        return True

    def to_str(self, val: Any) -> str:
        return str(val)

    def get_default(self) -> Any:
        raise NotSupportedError()

    def copy(self) -> "AnySpace":
        return AnySpace()

    def copy_value(self, val: Any) -> Any:
        raise NotSupportedError()

    def __eq__(self, o: "AnySpace") -> bool:
        return isinstance(o, AnySpace)

    def __str__(self) -> str:
        return "AnySpace()"

    @property
    def stype(self) -> SpaceTypes:
        return SpaceTypes.UNKNOWN

    @property
    def dtype(self):
        raise NotSupportedError()

    # --- stack
    def create_stack_space(self, length: int):
        from srl.base.spaces.multi import MultiSpace

        return MultiSpace([AnySpace() for _ in range(length)])

    def encode_stack(self, val: List[Any]) -> List[Any]:
        return val

    def get_onehot(self, x):
        raise NotSupportedError()

    # --------------------------------------
    # spaces
    # --------------------------------------
    def get_encode_type_list(self):
        priority_list = []
        exclude_list = []
        return priority_list, exclude_list

    # --- DiscreteSpace
    def create_encode_space_DiscreteSpace(self):
        return self

    def encode_to_space_DiscreteSpace(self, val: Any) -> int:
        return val

    def decode_from_space_DiscreteSpace(self, val: int) -> Any:
        return val

    # --- ArrayDiscreteSpace
    def create_encode_space_ArrayDiscreteSpace(self):
        return self

    def encode_to_space_ArrayDiscreteSpace(self, val: Any) -> List[int]:
        return val

    def decode_from_space_ArrayDiscreteSpace(self, val: List[int]) -> Any:
        return val

    # --- ContinuousSpace
    def create_encode_space_ContinuousSpace(self):
        return self

    def encode_to_space_ContinuousSpace(self, val: Any) -> float:
        return val

    def decode_from_space_ContinuousSpace(self, val: float) -> Any:
        return val

    # --- ArrayContinuousSpace
    def create_encode_space_ArrayContinuousListSpace(self):
        return self

    def encode_to_space_ArrayContinuousListSpace(self, val: Any) -> List[float]:
        return val

    def decode_from_space_ArrayContinuousListSpace(self, val: List[float]) -> Any:
        return val

    # --- np
    def create_encode_space_ArrayContinuousSpace(self, np_dtype):
        return self

    def encode_to_space_ArrayContinuousSpace(self, val: Any, space) -> np.ndarray:
        return val

    def decode_from_space_ArrayContinuousSpace(self, val: np.ndarray) -> Any:
        return val

    # --- Box
    def create_encode_space_Box(self, space_type, np_dtype):
        return self

    def encode_to_space_Box(self, val: Any, space) -> np.ndarray:
        return val

    def decode_from_space_Box(self, val: np.ndarray, space) -> Any:
        return val

    # --- TextSpace
    def create_encode_space_TextSpace(self):
        return self

    def encode_to_space_TextSpace(self, val: Any) -> str:
        return val

    def decode_from_space_TextSpace(self, val: str) -> Any:
        return val
