from abc import abstractmethod
from typing import Any, List, Tuple

import numpy as np
from srl.base.define import EnvAction, EnvInvalidAction, EnvObservationType, Info
from srl.base.env import EnvBase
from srl.base.env.base import SpaceBase


class SinglePlayEnv(EnvBase):

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

    @abstractmethod  # new method
    def reset_single(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod  # new method
    def step_single(self, action: EnvAction) -> Tuple[np.ndarray, float, bool, Info]:
        raise NotImplementedError()

    # new method(option)
    def get_invalid_actions_single(self) -> List[EnvInvalidAction]:
        return []

    @abstractmethod  # same parent
    def backup(self) -> Any:
        raise NotImplementedError()

    @abstractmethod  # same parent
    def restore(self, data: Any) -> None:
        raise NotImplementedError()

    # same parent(option)
    def action_to_str(self, action: EnvAction) -> str:
        return str(action)

    # --- inherit implementation(継承元の実装)

    @property
    def player_num(self) -> int:
        return 1

    def reset(self) -> Tuple[np.ndarray, List[int]]:
        return self.reset_single(), [0]

    def step(self, actions: List[EnvAction]) -> Tuple[np.ndarray, List[float], bool, List[int], Info]:
        n_state, reward, done, info = self.step_single(actions[0])
        return n_state, [reward], done, [0], info

    def get_next_player_indices(self) -> List[int]:
        return [0]

    def get_invalid_actions(self, player_index: int) -> List[EnvInvalidAction]:
        return self.get_invalid_actions_single()
