from typing import Any, List, Optional, Tuple

import numpy as np
from srl.base.define import Action, EnvObservationType, Info, InvalidAction
from srl.base.env import EnvBase
from srl.base.env.base import SpaceBase


class SinglePlayEnvWrapper(EnvBase):
    def __init__(self, env: EnvBase):
        self.env = env
        assert self.env.player_num == 1

    def close(self) -> None:
        self.env.close()

    @property
    def action_space(self) -> SpaceBase:
        return self.env.action_space

    @property
    def observation_space(self) -> SpaceBase:
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

    def reset(self) -> np.ndarray:
        state, next_player_indices = self.env.reset()
        return state

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Info]:
        next_state, rewards, done, next_player_indices, env_info = self.env.step([action])
        return next_state, rewards[0], done, env_info

    def get_next_player_indices(self) -> List[int]:
        return [0]

    def get_invalid_actions(self, player_index: int = 0) -> List[InvalidAction]:
        return self.env.get_invalid_actions(player_index)

    def render(self, *args):
        return self.env.render(*args)

    def backup(self) -> Any:
        return self.env.backup()

    def restore(self, state: Any) -> None:
        return self.env.restore(state)

    def action_to_str(self, action: Action) -> str:
        return self.env.action_to_str(action)

    def make_worker(self, name: str) -> Optional["srl.base.rl.base.RLWorker"]:
        return self.env.make_worker(name)

    def get_original_env(self) -> object:
        return self.env.get_original_env()

    def sample(self) -> Action:
        return self.env.sample([0])[0]
