import random
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvActionType, EnvObservationType
from srl.base.env import EnvBase


class SingleActionDiscrete(EnvBase):

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

    @abstractmethod  # new method
    def reset_single(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod  # new method
    def step_single(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError()

    # new method(option)
    def fetch_invalid_actions_single(self) -> List[int]:
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
        return 1

    def reset(self) -> Tuple[np.ndarray, List[int]]:
        return self.reset_single(), [0]

    def step(self, actions: List[Any]) -> Tuple[np.ndarray, List[float], bool, List[int], Dict[str, float]]:
        n_state, reward, done, info = self.step_single(actions[0])
        return n_state, [reward], done, [0], info

    def fetch_invalid_actions(self, player_index: int) -> List[int]:
        return self.fetch_invalid_actions_single()


class SingleActionContinuous(EnvBase):

    # --- inheritance target implementation(継承先の実装)

    @property
    @abstractmethod  # same parent
    def action_space(self) -> gym.spaces.Space:
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

    @abstractmethod  # new method
    def reset_single(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod  # new method
    def step_single(self, action: Any) -> Tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError()

    @abstractmethod  # same parent
    def backup(self) -> Any:
        raise NotImplementedError()

    @abstractmethod  # same parent
    def restore(self, data: Any) -> None:
        raise NotImplementedError()

    # --- inherit implementation(継承元の実装)

    @property
    def action_type(self) -> EnvActionType:
        return EnvActionType.CONTINUOUS

    @property
    def player_num(self) -> int:
        return 1

    def reset(self) -> Tuple[np.ndarray, List[int]]:
        return self.reset_single(), [0]

    def step(self, actions: List[Any]) -> Tuple[np.ndarray, List[float], bool, List[int], Dict[str, float]]:
        n_state, reward, done, info = self.step_single(actions[0])
        return n_state, [reward], done, [0], info

    def fetch_invalid_actions(self, player_index: int) -> List[int]:
        return []


if __name__ == "__main__":
    pass
