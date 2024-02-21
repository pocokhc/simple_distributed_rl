from abc import ABC, abstractmethod
from typing import Any, Generic, List, Tuple, TypeVar

import numpy as np

from srl.base.define import RLTypes
from srl.base.exception import NotSupportedError

T = TypeVar("T", int, List[int], float, List[float], np.ndarray, covariant=True)


class SpaceBase(ABC, Generic[T]):
    @abstractmethod
    def sample(self, mask: List[T] = []) -> T:
        """Returns a random value"""
        raise NotImplementedError()

    @abstractmethod
    def sanitize(self, val: Any) -> T:
        """Sanitize as much as possible"""
        raise NotImplementedError()

    @abstractmethod
    def check_val(self, val: Any) -> bool:
        """Check if val is a valid value for space"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def rl_type(self) -> RLTypes:
        """Return RLTypes"""
        raise NotImplementedError()

    @abstractmethod
    def get_default(self) -> T:
        """Return default value"""
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, __o: "SpaceBase") -> bool:
        raise NotImplementedError()

    # --- option
    def create_division_tbl(self, division_num: int) -> None:
        pass

    def get_valid_actions(self, mask: List[T] = []) -> List[T]:
        """Returns a valid actions"""
        raise NotSupportedError()

    # --------------------------------------
    # action discrete
    # --------------------------------------
    @property
    @abstractmethod
    def n(self) -> int:
        """discrete range"""
        raise NotImplementedError()

    @abstractmethod
    def encode_to_int(self, val) -> int:
        """SpaceVal -> int"""
        raise NotImplementedError()

    @abstractmethod
    def decode_from_int(self, val: int) -> T:
        """int -> SpaceVal"""
        raise NotImplementedError()

    # --------------------------------------
    # observation discrete
    # --------------------------------------
    @abstractmethod
    def encode_to_list_int(self, val) -> List[int]:
        """SpaceVal -> int -> np.ndarray"""
        raise NotImplementedError()

    @abstractmethod
    def decode_from_list_int(self, val: List[int]) -> T:
        """np.ndarray[int] -> SpaceVal"""
        raise NotImplementedError()

    # --------------------------------------
    # action continuous
    # --------------------------------------
    @property
    @abstractmethod
    def list_size(self) -> int:
        """continuous list length"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def list_low(self) -> List[float]:
        """continuous list length range"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def list_high(self) -> List[float]:
        """continuous list length range"""
        raise NotImplementedError()

    @abstractmethod
    def encode_to_list_float(self, val) -> List[float]:
        """SpaceVal -> list[float]"""
        raise NotImplementedError()

    @abstractmethod
    def decode_from_list_float(self, val: List[float]) -> T:
        """list[float] -> SpaceVal"""
        raise NotImplementedError()

    # --------------------------------------
    # observation continuous, image
    # --------------------------------------
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """numpy shape"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def low(self) -> np.ndarray:
        """numpy range"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def high(self) -> np.ndarray:
        """numpy range"""
        raise NotImplementedError()

    @abstractmethod
    def encode_to_np(self, val, dtype) -> np.ndarray:
        """SpaceVal -> np.ndarray"""
        raise NotImplementedError()

    @abstractmethod
    def decode_from_np(self, val: np.ndarray) -> T:
        """np.ndarray -> SpaceVal"""
        raise NotImplementedError()
