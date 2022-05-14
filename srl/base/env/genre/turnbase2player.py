from abc import abstractmethod
from typing import Any, List, Tuple

import numpy as np
from srl.base.define import Action, DiscreteAction, EnvObservationType, Info
from srl.base.env import EnvBase
from srl.base.env.base import SpaceBase


class TurnBase2Player(EnvBase):
    # --- inheritance target implementation(継承先の実装)

    @property
    @abstractmethod  # same parent
    def action_space(self) -> SpaceBase:
        raise NotImplementedError()

    @property
    @abstractmethod  # same parent
    def observation_space(self) -> SpaceBase:
        raise NotImplementedError()

    @property
    @abstractmethod  # same parent
    def observation_type(self) -> EnvObservationType:
        raise NotImplementedError()

    @property
    @abstractmethod  # same parent
    def max_episode_steps(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod  # new method
    def player_index(self) -> int:
        raise NotImplementedError()

    @abstractmethod  # new method
    def reset_turn(self) -> np.ndarray:
        # state
        raise NotImplementedError()

    @abstractmethod  # new method
    def step_turn(self, action: DiscreteAction) -> Tuple[np.ndarray, float, float, bool, Info]:
        # state, reward1, reward2, done, info
        raise NotImplementedError()

    # same parent(option)
    def get_invalid_actions(self, player_index: int) -> List[int]:
        return []

    @abstractmethod  # same parent
    def backup(self) -> Any:
        raise NotImplementedError()

    @abstractmethod  # same parent
    def restore(self, data: Any) -> None:
        raise NotImplementedError()

    # same parent(option)
    def action_to_str(self, action: Any) -> str:
        return str(action)

    # --- inherit implementation(継承元の実装)

    @property
    def player_num(self) -> int:
        return 2

    def reset(self) -> Tuple[np.ndarray, List[int]]:
        state = self.reset_turn()
        return state, [self.player_index]

    def step(self, actions: List[Action]) -> Tuple[np.ndarray, List[float], bool, List[int], Info]:
        n_s, reward1, reward2, done, info = self.step_turn(actions[0])
        return n_s, [reward1, reward2], done, [self.player_index], info

    def get_next_player_indices(self) -> List[int]:
        return [self.player_index]
