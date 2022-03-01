import json
import logging
from dataclasses import dataclass
from typing import Any

import gym
import gym.envs.registration
import gym.spaces
from srl.base.define import EnvObservationType
from srl.base.env import EnvBase

logger = logging.getLogger(__name__)

gym.envs.registration.register(
    id="OneRoad-v0",
    entry_point=__name__ + ":OneRoad",
    kwargs={"N": 10, "action_num": 2},  # 0.0009765625%
)
gym.envs.registration.register(
    id="OneRoad-v1",
    entry_point=__name__ + ":OneRoad",
    kwargs={"N": 30, "action_num": 16},
)


@dataclass
class OneRoad(EnvBase):

    N: int = 10
    action_num: int = 2

    def __post_init__(self):
        self._action_space = gym.spaces.Discrete(self.action_num)
        self._observation_space = gym.spaces.Box(low=0, high=self.N, shape=(1,))

    # override
    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    # override
    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    # override
    @property
    def observation_type(self) -> EnvObservationType:
        return EnvObservationType.DISCRETE

    # override
    @property
    def max_episode_steps(self) -> int:
        return int(self.N * 1.1)

    # override
    def reset(self) -> Any:
        self.player_pos = 0
        return [self.player_pos]

    # override
    def step(self, action: int) -> tuple[Any, float, bool, dict]:
        if action == 0:
            self.player_pos += 1
        else:
            self.player_pos = 0

        if self.player_pos == self.N:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        return [self.player_pos], reward, done, {}

    # override
    def fetch_valid_actions(self) -> list[int]:
        return [a for a in range(self.action_num)]

    # override
    def render(self, mode="human"):
        print(f"{self.player_pos} / {self.N}")
        return

    # override
    def backup(self) -> Any:
        return json.dumps(
            [
                self.player_pos,
            ]
        )

    # override
    def restore(self, data: Any) -> None:
        d = json.loads(data)
        self.player_pos = d[0]


if __name__ == "__main__":
    pass
