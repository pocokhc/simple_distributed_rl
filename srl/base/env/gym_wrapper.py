import pickle
from typing import Any, Dict, List, Tuple

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvActionType, EnvObservationType

from .base import EnvBase


class GymWrapper(EnvBase):
    def __init__(self, env_name: str):
        self.env: gym.Env = gym.make(env_name)

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.env.action_space

    @property
    def action_type(self) -> EnvActionType:
        return EnvActionType.UNKOWN

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.env.observation_space

    @property
    def observation_type(self) -> EnvObservationType:
        return EnvObservationType.UNKOWN

    @property
    def max_episode_steps(self) -> int:
        if hasattr(self.env, "_max_episode_steps"):
            return getattr(self.env, "_max_episode_steps")
        else:
            return 999_999

    @property
    def player_num(self) -> int:
        return 1

    def close(self) -> None:
        self.env.close()

    def reset(self) -> Tuple[np.ndarray, List[int]]:
        state = self.env.reset()
        return np.asarray(state), [0]

    def step(self, actions: List[Any]) -> Tuple[np.ndarray, List[float], bool, List[int], Dict[str, float]]:
        state, reward, done, info = self.env.step(actions[0])
        return np.asarray(state), [reward], done, [0], info

    def get_next_player_indecies(self) -> List[int]:
        return [0]

    def get_invalid_actions(self, player_index: int) -> List[int]:
        return []

    def render_terminal(self) -> None:
        print(self.env.render("ansi"))

    def render_gui(self) -> None:
        self.env.render("human")

    def render_rgb_array(self) -> np.ndarray:
        return np.asarray(self.env.render("rgb_array"))

    def backup(self) -> Any:
        return pickle.dumps(self.env)

    def restore(self, data: Any) -> None:
        self.env = pickle.loads(data)

    def get_original_env(self) -> object:
        return self.env


if __name__ == "__main__":
    pass
