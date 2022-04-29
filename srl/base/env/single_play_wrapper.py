from typing import Any, List, Optional, Tuple

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvActionType, EnvObservationType
from srl.base.env import EnvBase


class SinglePlayerWrapper(EnvBase):
    def __init__(self, env: EnvBase):
        self.env = env
        assert self.env.player_num == 1

    def close(self) -> None:
        self.env.close()

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.env.action_space

    @property
    def action_type(self) -> EnvActionType:
        return self.env.action_type

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.env.observation_space

    @property
    def observation_type(self) -> EnvObservationType:
        return self.env.observation_type

    @property
    def max_episode_steps(self) -> int:
        return self.env.max_episode_steps

    @property
    def player_num(self) -> int:
        return self.env.player_num

    def reset(self) -> Tuple[np.ndarray, List[int]]:
        state, next_player_indecies = self.env.reset()
        invalid_actions = self.fetch_invalid_actions()
        return state, invalid_actions

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, List[int], dict]:
        next_state, rewards, done, next_player_indices, env_info = self.env.step([action])
        invalid_actions = self.fetch_invalid_actions()
        return next_state, rewards[0], done, invalid_actions, env_info

    def fetch_invalid_actions(self) -> List[int]:
        return self.env.fetch_invalid_actions(0)

    def render(self, *args):
        return self.env.render(*args)

    def backup(self) -> Any:
        return self.env.backup()

    def restore(self, state: Any) -> None:
        return self.env.restore(state)

    def action_to_str(self, action: Any) -> str:
        return self.env.action_to_str(action)

    def make_worker(self, name: str) -> Optional["srl.base.rl.base.RLWorker"]:
        return self.env.make_worker(name)

    def get_original_env(self) -> object:
        return self.env.get_original_env()

    def sample(self) -> List[Any]:
        return self.env.sample([0])[0]


if __name__ == "__main__":
    pass
