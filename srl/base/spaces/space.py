from abc import ABC, abstractmethod
from typing import Any, Generic, List, Tuple, TypeVar

import numpy as np

from srl.base.define import RLActionType, RLObservationType, SpaceTypes
from srl.base.exception import NotSupportedError

TActSpace = TypeVar("TActSpace", bound="SpaceBase", covariant=True)
TActType = TypeVar("TActType", bound=RLActionType)
TObsSpace = TypeVar("TObsSpace", bound="SpaceBase", covariant=True)
TObsType = TypeVar("TObsType", bound=RLObservationType)

_T = TypeVar("_T")


class SpaceBase(ABC, Generic[_T]):
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
    # action discrete
    # --------------------------------------
    @property
    def int_size(self) -> int:
        """discrete range"""
        raise NotImplementedError()

    def encode_to_int(self, val: _T) -> int:
        """SpaceVal -> int"""
        raise NotImplementedError()

    def decode_from_int(self, val: int) -> _T:
        """int -> SpaceVal"""
        raise NotImplementedError()

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

    def encode_to_list_int(self, val: _T) -> List[int]:
        """SpaceVal -> int -> np.ndarray"""
        raise NotImplementedError()

    def decode_from_list_int(self, val: List[int]) -> _T:
        """np.ndarray[int] -> SpaceVal"""
        raise NotImplementedError()

    # --------------------------------------
    # action continuous
    # --------------------------------------
    @property
    def list_float_size(self) -> int:
        """continuous list length"""
        raise NotImplementedError()

    @property
    def list_float_low(self) -> List[float]:
        """continuous list length range"""
        raise NotImplementedError()

    @property
    def list_float_high(self) -> List[float]:
        """continuous list length range"""
        raise NotImplementedError()

    def encode_to_list_float(self, val: _T) -> List[float]:
        """SpaceVal -> list[float]"""
        raise NotImplementedError()

    def decode_from_list_float(self, val: List[float]) -> _T:
        """list[float] -> SpaceVal"""
        raise NotImplementedError()

    # --------------------------------------
    # observation continuous, image
    # --------------------------------------
    @property
    def np_shape(self) -> Tuple[int, ...]:
        """numpy shape"""
        raise NotImplementedError()

    @property
    def np_low(self) -> np.ndarray:
        """numpy range"""
        raise NotImplementedError()

    @property
    def np_high(self) -> np.ndarray:
        """numpy range"""
        raise NotImplementedError()

    def encode_to_np(self, val: _T, dtype) -> np.ndarray:
        """SpaceVal -> np.ndarray"""
        raise NotImplementedError()

    def decode_from_np(self, val: np.ndarray) -> _T:
        """np.ndarray -> SpaceVal"""
        raise NotImplementedError()

    # --------------------------------------
    # spaces
    # --------------------------------------
    def create_encode_space(self, space_name: str) -> "SpaceBase":
        raise NotImplementedError()

    def encode_to_space(self, val: _T, space: "SpaceBase") -> Any:
        # singledispatchmethodよりisinstanceの方が早い
        raise NotImplementedError()

    def decode_from_space(self, val: Any, space: "SpaceBase") -> _T:
        raise NotImplementedError()
