from abc import ABC, abstractmethod
from typing import Any, Generic, List, Tuple, TypeVar

import numpy as np
from srl.base.define import ContinuousAction, DiscreteAction, DiscreteSpaceType, RLObservation

T = TypeVar("T", int, List[int], float, List[float], np.ndarray, covariant=True)


class SpaceBase(ABC, Generic[T]):
    @abstractmethod
    def sample(self, invalid_actions: List[DiscreteSpaceType] = []) -> T:
        raise NotImplementedError()

    @abstractmethod
    def check_val(self, val: Any) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, __o: object) -> bool:
        return NotImplemented

    # --- action discrete
    @abstractmethod
    def get_action_discrete_info(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def action_discrete_encode(self, val: T) -> DiscreteAction:
        raise NotImplementedError()

    @abstractmethod
    def action_discrete_decode(self, val: DiscreteAction) -> T:
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
    def action_continuous_decode(self, val: ContinuousAction) -> T:
        raise NotImplementedError()

    # --- observation
    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError()  # shape

    @abstractmethod
    def observation_discrete_encode(self, val: T) -> RLObservation:
        raise NotImplementedError()

    @abstractmethod
    def observation_continuous_encode(self, val: T) -> RLObservation:
        raise NotImplementedError()
