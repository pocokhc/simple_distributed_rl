from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, List, Tuple, TypeVar

import numpy as np

from srl.base.define import RLBaseTypes, SpaceType, SpaceTypes
from srl.base.exception import NotSupportedError

if TYPE_CHECKING:
    from srl.base.spaces.array_continuous import ArrayContinuousSpace
    from srl.base.spaces.array_discrete import ArrayDiscreteSpace
    from srl.base.spaces.box import BoxSpace
    from srl.base.spaces.continuous import ContinuousSpace
    from srl.base.spaces.discrete import DiscreteSpace
    from srl.base.spaces.text import TextSpace

TActSpace = TypeVar("TActSpace", bound="SpaceBase", covariant=True)
TActType = TypeVar("TActType", bound=SpaceType)
TObsSpace = TypeVar("TObsSpace", bound="SpaceBase", covariant=True)
TObsType = TypeVar("TObsType", bound=SpaceType)

_T = TypeVar("_T")


class SpaceBase(ABC, Generic[_T]):
    @property
    @abstractmethod
    def name(self) -> str:
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

    @property
    @abstractmethod
    def stype(self) -> SpaceTypes:
        raise NotImplementedError()

    @property
    @abstractmethod
    def dtype(self):
        raise NotImplementedError()

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
        return self.stype == SpaceTypes.DISCRETE

    def is_continuous(self) -> bool:
        return self.stype == SpaceTypes.CONTINUOUS

    def get_onehot(self, x):
        raise NotImplementedError()

    # --------------------------------------
    # spaces
    # --------------------------------------
    def get_encode_type_list(self) -> Tuple[List[RLBaseTypes], List[RLBaseTypes]]:
        # return priority_list, exclude_list
        raise NotImplementedError()

    # @singledispatch より if の方が速い
    def create_encode_space(self, space_type: RLBaseTypes, np_dtype=np.float32) -> "SpaceBase":
        if space_type == RLBaseTypes.NONE:
            return self.create_encode_space_Self()
        elif space_type == RLBaseTypes.DISCRETE:
            return self.create_encode_space_DiscreteSpace()
        elif space_type == RLBaseTypes.ARRAY_DISCRETE:
            return self.create_encode_space_ArrayDiscreteSpace()
        elif space_type == RLBaseTypes.CONTINUOUS:
            return self.create_encode_space_ContinuousSpace()
        elif space_type == RLBaseTypes.ARRAY_CONTINUOUS:
            return self.create_encode_space_ArrayContinuousSpace()
        elif space_type in [
            RLBaseTypes.BOX,
            RLBaseTypes.GRAY_2ch,
            RLBaseTypes.GRAY_3ch,
            RLBaseTypes.COLOR,
            RLBaseTypes.IMAGE,
        ]:
            return self.create_encode_space_Box(space_type, np_dtype)
        elif space_type == RLBaseTypes.TEXT:
            return self.create_encode_space_TextSpace()
        # elif space_type == RLBaseTypes.MULTI:
        #    raise NotSupportedError()
        raise NotImplementedError(space_type)

    def encode_to_space(self, val: _T, space: "SpaceBase") -> Any:
        from srl.base.spaces.array_continuous import ArrayContinuousSpace
        from srl.base.spaces.array_discrete import ArrayDiscreteSpace
        from srl.base.spaces.box import BoxSpace
        from srl.base.spaces.continuous import ContinuousSpace
        from srl.base.spaces.discrete import DiscreteSpace
        from srl.base.spaces.text import TextSpace

        if isinstance(space, DiscreteSpace):
            return self.encode_to_space_DiscreteSpace(val)
        elif isinstance(space, ArrayDiscreteSpace):
            return self.encode_to_space_ArrayDiscreteSpace(val)
        elif isinstance(space, ContinuousSpace):
            return self.encode_to_space_ContinuousSpace(val)
        elif isinstance(space, ArrayContinuousSpace):
            return self.encode_to_space_ArrayContinuousSpace(val)
        elif isinstance(space, BoxSpace):
            return self.encode_to_space_Box(val, space)
        elif isinstance(space, TextSpace):
            return self.encode_to_space_TextSpace(val)
        return val

    def decode_from_space(self, val: Any, space: "SpaceBase") -> _T:
        from srl.base.spaces.array_continuous import ArrayContinuousSpace
        from srl.base.spaces.array_discrete import ArrayDiscreteSpace
        from srl.base.spaces.box import BoxSpace
        from srl.base.spaces.continuous import ContinuousSpace
        from srl.base.spaces.discrete import DiscreteSpace
        from srl.base.spaces.text import TextSpace

        if isinstance(space, DiscreteSpace):
            return self.decode_from_space_DiscreteSpace(val)
        elif isinstance(space, ArrayDiscreteSpace):
            return self.decode_from_space_ArrayDiscreteSpace(val)
        elif isinstance(space, ContinuousSpace):
            return self.decode_from_space_ContinuousSpace(val)
        elif isinstance(space, ArrayContinuousSpace):
            return self.decode_from_space_ArrayContinuousSpace(val)
        elif isinstance(space, BoxSpace):
            return self.decode_from_space_Box(val, space)
        elif isinstance(space, TextSpace):
            return self.decode_from_space_TextSpace(val)
        return val

    def create_encode_space_Self(self):
        return self.copy()

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

    # --- Box
    def create_encode_space_Box(self, space_type: RLBaseTypes, np_dtype) -> "BoxSpace":
        raise NotImplementedError()

    def encode_to_space_Box(self, val: _T, space: "BoxSpace") -> np.ndarray:
        raise NotImplementedError()

    def decode_from_space_Box(self, val: np.ndarray, space: "BoxSpace") -> _T:
        raise NotImplementedError()

    # --- TextSpace
    def create_encode_space_TextSpace(self) -> "TextSpace":
        raise NotImplementedError()

    def encode_to_space_TextSpace(self, val: _T) -> str:
        raise NotImplementedError()

    def decode_from_space_TextSpace(self, val: str) -> _T:
        raise NotImplementedError()
