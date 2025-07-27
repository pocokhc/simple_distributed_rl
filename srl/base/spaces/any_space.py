from typing import Any, List

import numpy as np

from srl.base.define import RLBaseTypes, SpaceTypes
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

    def equal_val(self, v1: Any, v2: Any) -> bool:
        return v1 == v2

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
    def get_encode_list(self):
        return [
            RLBaseTypes.DISCRETE,
            RLBaseTypes.ARRAY_DISCRETE,
            RLBaseTypes.CONTINUOUS,
            RLBaseTypes.ARRAY_CONTINUOUS,
            RLBaseTypes.NP_ARRAY,
            RLBaseTypes.NP_ARRAY_UNTYPED,
            RLBaseTypes.BOX,
            RLBaseTypes.BOX_UNTYPED,
            RLBaseTypes.TEXT,
            RLBaseTypes.MULTI,
        ]

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
    def create_encode_space_ArrayContinuousSpace(self):
        return self

    def encode_to_space_ArrayContinuousSpace(self, val: Any) -> List[float]:
        return val

    def decode_from_space_ArrayContinuousSpace(self, val: List[float]) -> Any:
        return val

    # --- NpArray
    def create_encode_space_NpArraySpace(self, dtype):
        return self

    def encode_to_space_NpArraySpace(self, val: Any, dtype) -> np.ndarray:
        return val

    def decode_from_space_NpArraySpace(self, val: np.ndarray) -> Any:
        return val

    # --- NpArrayUnTyped
    def create_encode_space_NpArrayUnTyped(self):
        return self

    def encode_to_space_NpArrayUnTyped(self, val: Any) -> np.ndarray:
        return val

    def decode_from_space_NpArrayUnTyped(self, val: np.ndarray) -> Any:
        return val

    # --- Box
    def create_encode_space_Box(self, dtype):
        return self

    def encode_to_space_Box(self, val: Any, dtype) -> np.ndarray:
        return val

    def decode_from_space_Box(self, val: np.ndarray) -> Any:
        return val

    # --- BoxUnTyped
    def create_encode_space_BoxUnTyped(self):
        return self

    def encode_to_space_BoxUnTyped(self, val: Any) -> np.ndarray:
        return val

    def decode_from_space_BoxUnTyped(self, val: np.ndarray) -> Any:
        return val

    # --- TextSpace
    def create_encode_space_TextSpace(self):
        return self

    def encode_to_space_TextSpace(self, val: Any) -> str:
        return val

    def decode_from_space_TextSpace(self, val: str) -> Any:
        return val

    # --- Multi
    def create_encode_space_MultiSpace(self):
        raise NotImplementedError()

    def encode_to_space_MultiSpace(self, val: Any) -> list:
        raise NotImplementedError()

    def decode_from_space_MultiSpace(self, val: list) -> Any:
        raise NotImplementedError()
