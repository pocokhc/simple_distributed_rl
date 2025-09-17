from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, List, Literal, TypeVar, overload

import numpy as np

from srl.base.define import RLBaseTypes, SpaceType, SpaceTypes
from srl.base.exception import NotSupportedError

if TYPE_CHECKING:
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


@dataclass
class SpaceEncodeOptions:
    cast: bool = False
    cast_dtype: Any = ""
    np_zero_start: bool = False
    np_norm_type: Literal["", "0to1", "-1to1"] = ""


class SpaceBase(ABC, Generic[_T]):
    def __init__(self):
        self.encode_space_type: RLBaseTypes = RLBaseTypes.NONE
        self.encode_options = SpaceEncodeOptions()
        self.encode_space: SpaceBase = None  # type: ignore

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

    @overload
    def set_encode_space(self, required_space_type: Literal[RLBaseTypes.NONE], options: SpaceEncodeOptions = SpaceEncodeOptions()) -> "SpaceBase": ...
    @overload
    def set_encode_space(self, required_space_type: Literal[RLBaseTypes.DISCRETE], options: SpaceEncodeOptions = SpaceEncodeOptions()) -> "DiscreteSpace": ...
    @overload
    def set_encode_space(self, required_space_type: Literal[RLBaseTypes.ARRAY_DISCRETE], options: SpaceEncodeOptions = SpaceEncodeOptions()) -> "ArrayDiscreteSpace": ...
    @overload
    def set_encode_space(self, required_space_type: Literal[RLBaseTypes.CONTINUOUS], options: SpaceEncodeOptions = SpaceEncodeOptions()) -> "ContinuousSpace": ...
    @overload
    def set_encode_space(self, required_space_type: Literal[RLBaseTypes.ARRAY_CONTINUOUS], options: SpaceEncodeOptions = SpaceEncodeOptions()) -> "ArrayContinuousSpace": ...
    @overload
    def set_encode_space(self, required_space_type: Literal[RLBaseTypes.NP_ARRAY], options: SpaceEncodeOptions = SpaceEncodeOptions()) -> "NpArraySpace": ...
    @overload
    def set_encode_space(self, required_space_type: Literal[RLBaseTypes.BOX], options: SpaceEncodeOptions = SpaceEncodeOptions()) -> "BoxSpace": ...
    @overload
    def set_encode_space(self, required_space_type: Literal[RLBaseTypes.TEXT], options: SpaceEncodeOptions = SpaceEncodeOptions()) -> "TextSpace": ...
    @overload
    def set_encode_space(self, required_space_type: Literal[RLBaseTypes.MULTI], options: SpaceEncodeOptions = SpaceEncodeOptions()) -> "MultiSpace": ...

    def set_encode_space(self, required_space_type: RLBaseTypes, options: SpaceEncodeOptions = SpaceEncodeOptions()) -> "SpaceBase":
        if options.cast_dtype == "":
            options.cast_dtype = np.float32
        if required_space_type == RLBaseTypes.NONE:
            space = self.copy()
        elif required_space_type == RLBaseTypes.DISCRETE:
            space = self._set_encode_space_DiscreteSpace(options)
        elif required_space_type == RLBaseTypes.ARRAY_DISCRETE:
            space = self._set_encode_space_ArrayDiscreteSpace(options)
        elif required_space_type == RLBaseTypes.CONTINUOUS:
            space = self._set_encode_space_ContinuousSpace(options)
        elif required_space_type == RLBaseTypes.ARRAY_CONTINUOUS:
            space = self._set_encode_space_ArrayContinuousSpace(options)
        elif required_space_type == RLBaseTypes.NP_ARRAY:
            space = self._set_encode_space_NpArraySpace(options)
        elif required_space_type == RLBaseTypes.BOX:
            space = self._set_encode_space_Box(options)
        elif required_space_type == RLBaseTypes.TEXT:
            space = self._set_encode_space_TextSpace(options)
        elif required_space_type == RLBaseTypes.MULTI:
            space = self._set_encode_space_MultiSpace(options)
        else:
            raise NotImplementedError(required_space_type)
        self.encode_space_type = required_space_type
        self.encode_options = options
        self.encode_space = space
        return space

    # @singledispatch より if の方が速い
    def encode_to_space(self, val: _T) -> Any:
        if self.encode_space_type == RLBaseTypes.DISCRETE:
            return self._encode_to_space_DiscreteSpace(val)
        elif self.encode_space_type == RLBaseTypes.ARRAY_DISCRETE:
            return self._encode_to_space_ArrayDiscreteSpace(val)
        elif self.encode_space_type == RLBaseTypes.CONTINUOUS:
            return self._encode_to_space_ContinuousSpace(val)
        elif self.encode_space_type == RLBaseTypes.ARRAY_CONTINUOUS:
            return self._encode_to_space_ArrayContinuousSpace(val)
        elif self.encode_space_type == RLBaseTypes.NP_ARRAY:
            return self._encode_to_space_NpArraySpace(val)
        elif self.encode_space_type == RLBaseTypes.BOX:
            return self._encode_to_space_Box(val)
        elif self.encode_space_type == RLBaseTypes.TEXT:
            return self._encode_to_space_TextSpace(val)
        elif self.encode_space_type == RLBaseTypes.MULTI:
            return self._encode_to_space_MultiSpace(val)
        return val

    def decode_from_space(self, val: Any) -> _T:
        if self.encode_space_type == RLBaseTypes.DISCRETE:
            return self._decode_from_space_DiscreteSpace(val)
        elif self.encode_space_type == RLBaseTypes.ARRAY_DISCRETE:
            return self._decode_from_space_ArrayDiscreteSpace(val)
        elif self.encode_space_type == RLBaseTypes.CONTINUOUS:
            return self._decode_from_space_ContinuousSpace(val)
        elif self.encode_space_type == RLBaseTypes.ARRAY_CONTINUOUS:
            return self._decode_from_space_ArrayContinuousSpace(val)
        elif self.encode_space_type == RLBaseTypes.NP_ARRAY:
            return self._decode_from_space_NpArraySpace(val)
        elif self.encode_space_type == RLBaseTypes.BOX:
            return self._decode_from_space_Box(val)
        elif self.encode_space_type == RLBaseTypes.TEXT:
            return self._decode_from_space_TextSpace(val)
        elif self.encode_space_type == RLBaseTypes.MULTI:
            return self._decode_from_space_MultiSpace(val)
        return val

    # --- DiscreteSpace
    def _set_encode_space_DiscreteSpace(self, options: SpaceEncodeOptions) -> "DiscreteSpace":
        raise NotImplementedError()

    def _encode_to_space_DiscreteSpace(self, val: _T) -> int:
        raise NotImplementedError()

    def _decode_from_space_DiscreteSpace(self, val: int) -> _T:
        raise NotImplementedError()

    # --- ArrayDiscreteSpace
    def _set_encode_space_ArrayDiscreteSpace(self, options: SpaceEncodeOptions) -> "ArrayDiscreteSpace":
        raise NotImplementedError()

    def _encode_to_space_ArrayDiscreteSpace(self, val: _T) -> List[int]:
        raise NotImplementedError()

    def _decode_from_space_ArrayDiscreteSpace(self, val: List[int]) -> _T:
        raise NotImplementedError()

    # --- ContinuousSpace
    def _set_encode_space_ContinuousSpace(self, options: SpaceEncodeOptions) -> "ContinuousSpace":
        raise NotImplementedError()

    def _encode_to_space_ContinuousSpace(self, val: _T) -> float:
        raise NotImplementedError()

    def _decode_from_space_ContinuousSpace(self, val: float) -> _T:
        raise NotImplementedError()

    # --- ArrayContinuousSpace
    def _set_encode_space_ArrayContinuousSpace(self, options: SpaceEncodeOptions) -> "ArrayContinuousSpace":
        raise NotImplementedError()

    def _encode_to_space_ArrayContinuousSpace(self, val: _T) -> List[float]:
        raise NotImplementedError()

    def _decode_from_space_ArrayContinuousSpace(self, val: List[float]) -> _T:
        raise NotImplementedError()

    # --- NpArray
    def _set_encode_space_NpArraySpace(self, options: SpaceEncodeOptions) -> "NpArraySpace":
        raise NotImplementedError()

    def _encode_to_space_NpArraySpace(self, val: _T) -> np.ndarray:
        raise NotImplementedError()

    def _decode_from_space_NpArraySpace(self, val: np.ndarray) -> _T:
        raise NotImplementedError()

    # --- Box
    def _set_encode_space_Box(self, options: SpaceEncodeOptions) -> "BoxSpace":
        raise NotImplementedError()

    def _encode_to_space_Box(self, val: _T) -> np.ndarray:
        raise NotImplementedError()

    def _decode_from_space_Box(self, val: np.ndarray) -> _T:
        raise NotImplementedError()

    # --- TextSpace
    def _set_encode_space_TextSpace(self, options: SpaceEncodeOptions) -> "TextSpace":
        raise NotImplementedError()

    def _encode_to_space_TextSpace(self, val: _T) -> str:
        raise NotImplementedError()

    def _decode_from_space_TextSpace(self, val: str) -> _T:
        raise NotImplementedError()

    # --- Multi
    def _set_encode_space_MultiSpace(self, options: SpaceEncodeOptions) -> "MultiSpace":
        raise NotImplementedError()

    def _encode_to_space_MultiSpace(self, val: _T) -> list:
        raise NotImplementedError()

    def _decode_from_space_MultiSpace(self, val: list) -> _T:
        raise NotImplementedError()
