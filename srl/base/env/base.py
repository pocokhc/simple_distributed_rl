import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np
from srl.base.define import (
    ContinuousAction,
    DiscreteAction,
    DiscreteSpaceType,
    EnvAction,
    EnvInvalidAction,
    EnvObservation,
    EnvObservationType,
    Info,
    RenderType,
    RLObservation,
    SpaceType,
)

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    name: str
    kwargs: Dict = field(default_factory=dict)

    # gym
    gym_prediction_by_simulation: bool = True


class SpaceBase(ABC):
    @abstractmethod
    def sample(self, invalid_actions: List[DiscreteSpaceType] = []) -> SpaceType:
        raise NotImplementedError()

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


class EnvBase(ABC):
    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        pass

    # --- action
    @property
    @abstractmethod
    def action_space(self) -> SpaceBase:
        raise NotImplementedError()

    # --- observation
    @property
    @abstractmethod
    def observation_space(self) -> SpaceBase:
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_type(self) -> EnvObservationType:
        raise NotImplementedError()

    # --- properties
    @property
    @abstractmethod
    def max_episode_steps(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def player_num(self) -> int:
        raise NotImplementedError()

    # --- implement functions
    @abstractmethod
    def reset(self) -> Tuple[EnvObservation, List[int]]:
        """
        return (
            init_state,
            next_player_indices,
        )
        """
        raise NotImplementedError()

    @abstractmethod
    def step(self, actions: List[EnvAction]) -> Tuple[EnvObservation, List[float], bool, List[int], Info]:
        """
        return (
            next_state,
            [
                player1 reward,
                player2 reward,
                ...
            ],
            done,
            (next_players_info) [
                "player_index": int,
                "invalid_actions": List[int],
            ],
            info,
        )
        """
        raise NotImplementedError()

    @abstractmethod
    def get_next_player_indices(self) -> List[int]:
        raise NotImplementedError()

    @abstractmethod
    def get_invalid_actions(self, player_index: int) -> List[EnvInvalidAction]:
        raise NotImplementedError()

    def sample(self, next_player_indices) -> List[EnvAction]:
        return [self.action_space.sample(self.get_invalid_actions(i)) for i in next_player_indices]

    def render(self, mode: RenderType = RenderType.Terminal, **kwargs) -> Any:
        try:
            if mode == RenderType.Terminal:
                return self.render_terminal(**kwargs)
            elif mode == RenderType.GUI:
                return self.render_gui(**kwargs)
            elif mode == RenderType.RGB_Array:
                return self.render_rgb_array(**kwargs)
        except NotImplementedError:
            # logger.warn(f"render NotImplementedError({mode})")
            pass

    def render_terminal(self, **kwargs) -> None:
        raise NotImplementedError()

    def render_gui(self, **kwargs) -> None:
        raise NotImplementedError()

    def render_rgb_array(self, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def backup(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def restore(self, data: Any) -> None:
        raise NotImplementedError()

    def copy(self):
        env = self.__class__()
        env.restore(self.backup())
        return env

    def action_to_str(self, action: EnvAction) -> str:
        return str(action)

    def make_worker(self, name: str) -> Optional["srl.base.rl.base.RLWorker"]:
        return None

    def get_original_env(self) -> object:
        return self
