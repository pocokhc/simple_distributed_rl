from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvActionType, EnvObservationType
from srl.base.env import EnvBase


class TurnBase2PlayerActionDiscrete(EnvBase):
    # --- inheritance target implementation(継承先の実装)

    @property
    @abstractmethod  # new method
    def action_num(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod  # same parent
    def observation_space(self) -> gym.spaces.Space:
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
    def step_turn(self, action: int) -> Tuple[np.ndarray, float, float, bool, dict]:
        # state, reward1, reward2, done, info
        raise NotImplementedError()

    # same parent(option)
    def fetch_invalid_actions(self, player_index: int) -> List[int]:
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
    def action_space(self) -> gym.spaces.Space:
        assert self.action_num >= 2
        return gym.spaces.Discrete(self.action_num)

    @property
    def action_type(self) -> EnvActionType:
        return EnvActionType.DISCRETE

    @property
    def player_num(self) -> int:
        return 2

    def reset(self) -> Tuple[np.ndarray, List[int]]:
        state = self.reset_turn()
        return state, [self.player_index]

    def step(self, actions: List[Any]) -> Tuple[np.ndarray, List[float], bool, List[int], Dict[str, float]]:
        n_s, reward1, reward2, done, info = self.step_turn(actions[0])
        return n_s, [reward1, reward2], done, [self.player_index], info


if __name__ == "__main__":
    pass
