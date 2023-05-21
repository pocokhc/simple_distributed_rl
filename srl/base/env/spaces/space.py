from abc import ABC, abstractmethod
from typing import Any, Generic, List, Tuple, TypeVar

import numpy as np

from srl.base.define import (
    ContinuousActionType,
    DiscreteActionType,
    InvalidActionsType,
    RLActionTypes,
    RLObservationType,
    RLObservationTypes,
)

T = TypeVar("T", int, List[int], float, List[float], np.ndarray, covariant=True)


class SpaceBase(ABC, Generic[T]):
    @abstractmethod
    def sample(self, invalid_actions: InvalidActionsType = []) -> T:
        """ランダムな値を返す"""
        raise NotImplementedError()

    @abstractmethod
    def convert(self, val: Any) -> T:
        """可能な限り変換する"""
        raise NotImplementedError()

    @abstractmethod
    def check_val(self, val: Any) -> bool:
        """val が space の値として妥当か確認"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def rl_action_type(self) -> RLActionTypes:
        """RLActionTypes を返す"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def rl_observation_type(self) -> RLObservationTypes:
        """RLObservationTypes を返す"""
        raise NotImplementedError()

    @abstractmethod
    def get_default(self) -> T:
        """return default value"""
        return NotImplemented

    @abstractmethod
    def __eq__(self, __o: "SpaceBase") -> bool:
        return NotImplemented

    # --- action discrete
    @abstractmethod
    def get_action_discrete_info(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def action_discrete_encode(self, val) -> DiscreteActionType:
        raise NotImplementedError()

    @abstractmethod
    def action_discrete_decode(self, val: DiscreteActionType) -> T:
        raise NotImplementedError()

    # --- action continuous
    @abstractmethod
    def get_action_continuous_info(self) -> Tuple[int, np.ndarray, np.ndarray]:
        raise NotImplementedError()  # n, low, high

    # not use
    # @abstractmethod
    # def action_continuous_encode(self, val: T) -> ContinuousAction:
    #    raise NotImplementedError()

    @abstractmethod
    def action_continuous_decode(self, val: ContinuousActionType) -> T:
        raise NotImplementedError()

    # --- observation
    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError()  # shape

    @abstractmethod
    def observation_discrete_encode(self, val) -> RLObservationType:
        raise NotImplementedError()

    @abstractmethod
    def observation_continuous_encode(self, val) -> RLObservationType:
        raise NotImplementedError()
