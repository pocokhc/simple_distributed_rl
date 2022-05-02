import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvActionType, EnvObservationType, RenderType


class EnvBase(ABC):
    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        pass

    # --- propeties

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Space:
        raise NotImplementedError()

    @property
    @abstractmethod
    def action_type(self) -> EnvActionType:
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Space:
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_type(self) -> EnvObservationType:
        raise NotImplementedError()

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
    def reset(self) -> Tuple[np.ndarray, List[int]]:
        """
        return (
            init_state,
            next_player_indecies,
        )
        """
        raise NotImplementedError()

    @abstractmethod
    def step(self, actions: List[Any]) -> Tuple[np.ndarray, List[float], bool, List[int], Dict[str, float]]:
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
    def get_next_player_indecies(self) -> List[int]:
        raise NotImplementedError()

    @abstractmethod
    def get_invalid_actions(self, player_index: int) -> List[int]:
        raise NotImplementedError()

    def sample(self, next_player_indices) -> List[Any]:
        if self.action_type == EnvActionType.DISCRETE:
            actions = []
            for i in next_player_indices:
                invalid_actions = self.get_invalid_actions(i)
                _actions = [a for a in range(self.action_space.n) if a not in invalid_actions]
                action = random.choice(_actions)
                actions.append(action)
            return actions
        else:
            return [self.action_space.sample() for _ in range(self.player_num)]

    def render(self, mode: RenderType = RenderType.Terminal, **kwargs) -> Any:
        if mode == RenderType.Terminal:
            return self.render_terminal(**kwargs)
        elif mode == RenderType.GUI:
            return self.render_gui(**kwargs)
        elif mode == RenderType.RGB_Array:
            return self.render_rgb_array(**kwargs)

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

    def action_to_str(self, action: Any) -> str:
        return str(action)

    def make_worker(self, name: str) -> Optional["srl.base.rl.base.RLWorker"]:
        return None

    def get_original_env(self) -> object:
        return self


if __name__ == "__main__":
    pass
