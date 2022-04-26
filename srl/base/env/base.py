import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

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
    def reset(self) -> Tuple[List[np.ndarray], List[int]]:
        """
        return (
            [
                player1 init_state,
                player2 init_state,
                ...
            ],
            start player index list,
        )
        """
        raise NotImplementedError()

    @abstractmethod
    def step(
        self, actions: List[Any], player_indexes: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[int], bool, dict]:
        """
        return (
            [
                player1 next_state,
                player2 next_state,
                ...
            ],
            [
                player1 reward,
                player2 reward,
                ...
            ],
            next player index list,
            done,
            info,
        )
        """
        raise NotImplementedError()

    @abstractmethod
    def fetch_invalid_actions(self) -> List[List[int]]:
        """
        return [
            player1 invalid_actions,
            player2 invalid_actions,
            ...
        ]
        """
        raise NotImplementedError()

    def sample(self) -> List[Any]:
        if self.action_type == EnvActionType.DISCRETE:
            actions = []
            for invalid in self.fetch_invalid_actions():
                _actions = [a for a in range(self.action_space.n) if a not in invalid]
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

    def action_to_str(self, action: Any) -> str:
        return str(action)

    def make_worker(self, name: str) -> Optional["srl.base.rl.base.RLWorker"]:
        return None

    def get_original_env(self) -> object:
        return self


if __name__ == "__main__":
    pass
