from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from srl.base.define import ContinuousAction, DiscreteAction, DiscreteSpaceType, RLObservation, SpaceType


class SpaceBase(ABC):
    @abstractmethod
    def sample(self, invalid_actions: List[DiscreteSpaceType] = []) -> SpaceType:
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, __o: object) -> bool:
        return NotImplemented

    # --- action discrete
    @abstractmethod
    def get_action_discrete_info(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def action_discrete_encode(self, val: SpaceType) -> DiscreteAction:
        raise NotImplementedError()

    @abstractmethod
    def action_discrete_decode(self, val: DiscreteAction) -> SpaceType:
        raise NotImplementedError()

    # --- action continuous
    @abstractmethod
    def get_action_continuous_info(self) -> Tuple[int, np.ndarray, np.ndarray]:
        raise NotImplementedError()  # n, low, high

    # not use
    # @abstractmethod
    # def action_continuous_encode(self, val: SpaceType) -> ContinuousAction:
    #    raise NotImplementedError()

    @abstractmethod
    def action_continuous_decode(self, val: ContinuousAction) -> SpaceType:
        raise NotImplementedError()

    # --- observation discrete
    @abstractmethod
    def get_observation_discrete_info(self) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray]:
        raise NotImplementedError()  # shape, low, high

    @abstractmethod
    def observation_discrete_encode(self, val: SpaceType) -> RLObservation:
        raise NotImplementedError()

    # not use
    # @abstractmethod
    # def observation_discrete_decode(self, val: RLObservation) -> SpaceType:
    #    raise NotImplementedError()

    # --- observation continuous
    @abstractmethod
    def get_observation_continuous_info(self) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray]:
        raise NotImplementedError()  # shape, low, high

    @abstractmethod
    def observation_continuous_encode(self, val: SpaceType) -> RLObservation:
        raise NotImplementedError()

    # not use
    # @abstractmethod
    # def observation_continuous_decode(self, val: RLObservation) -> SpaceType:
    #    raise NotImplementedError()
