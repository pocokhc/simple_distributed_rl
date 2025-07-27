from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, List, Optional, TypeVar

import numpy as np

from srl.base.define import RLBaseTypes, SpaceType, SpaceTypes
from srl.base.exception import NotSupportedError

if TYPE_CHECKING:
    from srl.base.rl.config import RLConfig
    from srl.base.spaces.array_continuous import ArrayContinuousSpace
    from srl.base.spaces.array_discrete import ArrayDiscreteSpace
    from srl.base.spaces.box import BoxSpace
    from srl.base.spaces.continuous import ContinuousSpace
    from srl.base.spaces.discrete import DiscreteSpace
    from srl.base.spaces.multi import MultiSpace
    from srl.base.spaces.np_array import NpArraySpace
    from srl.base.spaces.text import TextSpace

TActSpace = TypeVar("TActSpace", bound="SpaceBase", covariant=True)
TActType = TypeVar("TActType", bound=SpaceType)
TObsSpace = TypeVar("TObsSpace", bound="SpaceBase", covariant=True)
TObsType = TypeVar("TObsType", bound=SpaceType)

_T = TypeVar("_T")


class SpaceBase(ABC, Generic[_T]):
    def __init__(self):
        self.encode_space_type: RLBaseTypes = RLBaseTypes.NONE

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def stype(self) -> SpaceTypes:
        raise NotImplementedError()

    @property
    @abstractmethod
    def dtype(self):
        raise NotImplementedError()

    @abstractmethod
    def sample(self, mask: List[_T] = []) -> _T:
        """Returns a random value"""
        raise NotImplementedError()

    @abstractmethod
    def sanitize(self, val: Any) -> _T:
        """Sanitize as much as possible"""
        raise NotImplementedError()

    @abstractmethod
    def check_val(self, val: Any) -> bool:
        """Check if val is a valid value for space"""
        raise NotImplementedError()

    @abstractmethod
    def to_str(self, val: _T) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_default(self) -> _T:
        """Return default value"""
        raise NotImplementedError()

    @abstractmethod
    def copy(self, **kwargs) -> "SpaceBase":
        """引数はコンストラクタを上書き"""
        raise NotImplementedError()

    @abstractmethod
    def copy_value(self, v: _T) -> _T:
        raise NotImplementedError()

    @abstractmethod
    def equal_val(self, v1: _T, v2: _T) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, __o: "SpaceBase") -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    # --- option
    def create_division_tbl(
        self,
        division_num: int,
        max_size: int = 100_000,
        max_byte: int = 1024 * 1024 * 1024,
    ) -> None:
        pass

    def get_valid_actions(self, masks: List[_T] = []) -> List[_T]:
        """Returns a valid actions"""
        raise NotSupportedError()

    # --- stack
    @abstractmethod
    def create_stack_space(self, length: int) -> "SpaceBase":
        raise NotImplementedError()

    @abstractmethod
    def encode_stack(self, val: List[_T]):
        raise NotImplementedError()

    # --- utils
    def is_value(self) -> bool:
        return self.stype in [
            SpaceTypes.DISCRETE,
            SpaceTypes.CONTINUOUS,
        ]

    def is_image(self, in_image: bool = True) -> bool:
        if in_image:
            return self.stype in [
                SpaceTypes.GRAY_2ch,
                SpaceTypes.GRAY_3ch,
                SpaceTypes.COLOR,
                SpaceTypes.IMAGE,
            ]
        else:
            return self.stype in [
                SpaceTypes.GRAY_2ch,
                SpaceTypes.GRAY_3ch,
                SpaceTypes.COLOR,
            ]

    def is_multi(self) -> bool:
        return self.stype == SpaceTypes.MULTI

    def is_discrete(self) -> bool:
        return "int" in str(self.dtype)

    def is_continuous(self) -> bool:
        return "float" in str(self.dtype)

    def get_onehot(self, x):
        raise NotImplementedError()

    # --------------------------------------
    # spaces
    # --------------------------------------
    def get_encode_list(self) -> List[RLBaseTypes]:
        raise NotImplementedError()

    # @singledispatch より if の方が速い
    def create_encode_space(self, required_space_type: RLBaseTypes, rl_config: Optional["RLConfig"] = None) -> "SpaceBase":
        if rl_config is None:
            dtype = np.float32
        else:
            dtype = rl_config.get_dtype("np")

        if required_space_type == RLBaseTypes.NONE:
            space = self.copy()
        elif required_space_type == RLBaseTypes.DISCRETE:
            space = self.create_encode_space_DiscreteSpace()
        elif required_space_type == RLBaseTypes.ARRAY_DISCRETE:
            space = self.create_encode_space_ArrayDiscreteSpace()
        elif required_space_type == RLBaseTypes.CONTINUOUS:
            space = self.create_encode_space_ContinuousSpace()
        elif required_space_type == RLBaseTypes.ARRAY_CONTINUOUS:
            space = self.create_encode_space_ArrayContinuousSpace()
        elif required_space_type == RLBaseTypes.NP_ARRAY:
            space = self.create_encode_space_NpArraySpace(dtype)
        elif required_space_type == RLBaseTypes.NP_ARRAY_UNTYPED:
            space = self.create_encode_space_NpArrayUnTyped()
        elif required_space_type == RLBaseTypes.BOX:
            space = self.create_encode_space_Box(dtype)
        elif required_space_type == RLBaseTypes.BOX_UNTYPED:
            space = self.create_encode_space_BoxUnTyped()
        elif required_space_type == RLBaseTypes.TEXT:
            space = self.create_encode_space_TextSpace()
        elif required_space_type == RLBaseTypes.MULTI:
            space = self.create_encode_space_MultiSpace()
        else:
            raise NotImplementedError(required_space_type)
        space.encode_space_type = required_space_type
        return space

    def encode_to_space(self, val: _T, space: "SpaceBase") -> Any:
        if space.encode_space_type == RLBaseTypes.DISCRETE:
            return self.encode_to_space_DiscreteSpace(val)
        elif space.encode_space_type == RLBaseTypes.ARRAY_DISCRETE:
            return self.encode_to_space_ArrayDiscreteSpace(val)
        elif space.encode_space_type == RLBaseTypes.CONTINUOUS:
            return self.encode_to_space_ContinuousSpace(val)
        elif space.encode_space_type == RLBaseTypes.ARRAY_CONTINUOUS:
            return self.encode_to_space_ArrayContinuousSpace(val)
        elif space.encode_space_type == RLBaseTypes.NP_ARRAY:
            return self.encode_to_space_NpArraySpace(val, space.dtype)
        elif space.encode_space_type == RLBaseTypes.NP_ARRAY_UNTYPED:
            return self.encode_to_space_NpArrayUnTyped(val)
        elif space.encode_space_type == RLBaseTypes.BOX:
            return self.encode_to_space_Box(val, space.dtype)
        elif space.encode_space_type == RLBaseTypes.BOX_UNTYPED:
            return self.encode_to_space_BoxUnTyped(val)
        elif space.encode_space_type == RLBaseTypes.TEXT:
            return self.encode_to_space_TextSpace(val)
        elif space.encode_space_type == RLBaseTypes.MULTI:
            return self.encode_to_space_MultiSpace(val)
        return val

    def decode_from_space(self, val: Any, space: "SpaceBase") -> _T:
        if space.encode_space_type == RLBaseTypes.DISCRETE:
            return self.decode_from_space_DiscreteSpace(val)
        elif space.encode_space_type == RLBaseTypes.ARRAY_DISCRETE:
            return self.decode_from_space_ArrayDiscreteSpace(val)
        elif space.encode_space_type == RLBaseTypes.CONTINUOUS:
            return self.decode_from_space_ContinuousSpace(val)
        elif space.encode_space_type == RLBaseTypes.ARRAY_CONTINUOUS:
            return self.decode_from_space_ArrayContinuousSpace(val)
        elif space.encode_space_type == RLBaseTypes.NP_ARRAY:
            return self.decode_from_space_NpArraySpace(val)
        elif space.encode_space_type == RLBaseTypes.NP_ARRAY_UNTYPED:
            return self.decode_from_space_NpArrayUnTyped(val)
        elif space.encode_space_type == RLBaseTypes.BOX:
            return self.decode_from_space_Box(val)
        elif space.encode_space_type == RLBaseTypes.BOX_UNTYPED:
            return self.decode_from_space_BoxUnTyped(val)
        elif space.encode_space_type == RLBaseTypes.TEXT:
            return self.decode_from_space_TextSpace(val)
        elif space.encode_space_type == RLBaseTypes.MULTI:
            return self.decode_from_space_MultiSpace(val)
        return val

    # --- DiscreteSpace
    def create_encode_space_DiscreteSpace(self) -> "DiscreteSpace":
        raise NotImplementedError()

    def encode_to_space_DiscreteSpace(self, val: _T) -> int:
        raise NotImplementedError()

    def decode_from_space_DiscreteSpace(self, val: int) -> _T:
        raise NotImplementedError()

    # --- ArrayDiscreteSpace
    def create_encode_space_ArrayDiscreteSpace(self) -> "ArrayDiscreteSpace":
        raise NotImplementedError()

    def encode_to_space_ArrayDiscreteSpace(self, val: _T) -> List[int]:
        raise NotImplementedError()

    def decode_from_space_ArrayDiscreteSpace(self, val: List[int]) -> _T:
        raise NotImplementedError()

    # --- ContinuousSpace
    def create_encode_space_ContinuousSpace(self) -> "ContinuousSpace":
        raise NotImplementedError()

    def encode_to_space_ContinuousSpace(self, val: _T) -> float:
        raise NotImplementedError()

    def decode_from_space_ContinuousSpace(self, val: float) -> _T:
        raise NotImplementedError()

    # --- ArrayContinuousSpace
    def create_encode_space_ArrayContinuousSpace(self) -> "ArrayContinuousSpace":
        raise NotImplementedError()

    def encode_to_space_ArrayContinuousSpace(self, val: _T) -> List[float]:
        raise NotImplementedError()

    def decode_from_space_ArrayContinuousSpace(self, val: List[float]) -> _T:
        raise NotImplementedError()

    # --- NpArray
    def create_encode_space_NpArraySpace(self, dtype) -> "NpArraySpace":
        raise NotImplementedError()

    def encode_to_space_NpArraySpace(self, val: _T, dtype) -> np.ndarray:
        raise NotImplementedError()

    def decode_from_space_NpArraySpace(self, val: np.ndarray) -> _T:
        raise NotImplementedError()

    # --- NpArrayUnTyped
    def create_encode_space_NpArrayUnTyped(self) -> "NpArraySpace":
        raise NotImplementedError()

    def encode_to_space_NpArrayUnTyped(self, val: _T) -> np.ndarray:
        raise NotImplementedError()

    def decode_from_space_NpArrayUnTyped(self, val: np.ndarray) -> _T:
        raise NotImplementedError()

    # --- Box
    def create_encode_space_Box(self, dtype) -> "BoxSpace":
        raise NotImplementedError()

    def encode_to_space_Box(self, val: _T, dtype) -> np.ndarray:
        raise NotImplementedError()

    def decode_from_space_Box(self, val: np.ndarray) -> _T:
        raise NotImplementedError()

    # --- BoxUnTyped
    def create_encode_space_BoxUnTyped(self) -> "BoxSpace":
        raise NotImplementedError()

    def encode_to_space_BoxUnTyped(self, val: _T) -> np.ndarray:
        raise NotImplementedError()

    def decode_from_space_BoxUnTyped(self, val: np.ndarray) -> _T:
        raise NotImplementedError()

    # --- TextSpace
    def create_encode_space_TextSpace(self) -> "TextSpace":
        raise NotImplementedError()

    def encode_to_space_TextSpace(self, val: _T) -> str:
        raise NotImplementedError()

    def decode_from_space_TextSpace(self, val: str) -> _T:
        raise NotImplementedError()

    # --- Multi
    def create_encode_space_MultiSpace(self) -> "MultiSpace":
        raise NotImplementedError()

    def encode_to_space_MultiSpace(self, val: _T) -> list:
        raise NotImplementedError()

    def decode_from_space_MultiSpace(self, val: list) -> _T:
        raise NotImplementedError()
