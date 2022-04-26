from abc import abstractmethod
from typing import Any, List, Tuple

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvActionType, EnvObservationType
from srl.base.env import EnvBase


class TurnBase2PlayerActionDiscrete(EnvBase):
    def __init__(self):
        self._player_index = 0

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
    def reset_turn(self) -> Tuple[np.ndarray, np.ndarray]:
        # state1, state2
        raise NotImplementedError()

    @abstractmethod  # new method
    def step_turn(self, action: int) -> Tuple[np.ndarray, np.ndarray, float, float, bool, dict]:
        # state1, state2, reward1, reward2, done, info
        raise NotImplementedError()

    # new method(option)
    def fetch_invalid_actions_turn(self) -> Tuple[List[int], List[int]]:
        return [], []

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

    def reset(self) -> Tuple[List[np.ndarray], List[int]]:
        state1, state2 = self.reset_turn()
        assert 0 <= self.player_index <= 1
        return [state1, state2], [self.player_index]

    def step(
        self, actions: List[Any], player_indexes: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[int], bool, dict]:
        n_s1, n_s2, reward1, reward2, done, info = self.step_turn(actions[self.player_index])
        return [n_s1, n_s2], [reward1, reward2], [self.player_index], done, info

    def fetch_invalid_actions(self) -> List[List[int]]:
        va1, va2 = self.fetch_invalid_actions_turn()
        return [va1, va2]


if __name__ == "__main__":
    pass
