import enum
import json
import logging
import random
from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import gym
import gym.envs.registration
import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType
from srl.base.env import EnvBase

logger = logging.getLogger(__name__)


gym.envs.registration.register(
    id="MyEnv-v0",
    entry_point=__name__ + ":MyEnv",
    kwargs={
        "move_reward": -0.04,
    },
)


class Action(enum.Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


@dataclass
class MyEnv(EnvBase):

    move_reward: float = -0.04

    def __post_init__(self):

        self.base_field = [
            [0, 0, 0, 1],
            [0, 9, 0, -1],
            [0, 0, 0, 0],
        ]
        self.H = 3
        self.W = 4

        self._action_space = gym.spaces.Discrete(len(Action))
        self._observation_space = gym.spaces.Box(
            low=0,
            high=np.asarray([self.W, self.H]),
            shape=(2,),
        )

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
        return 100

    # override
    def fetch_valid_actions(self) -> List[int]:
        return [e.value for e in Action]

    # override
    def reset(self) -> Any:
        self.player_pos = [0, 2]
        return tuple(self.player_pos)

    # override
    def step(self, action_: int) -> Tuple[Any, float, bool, dict]:
        action = Action(action_)

        next_player_pos = self.player_pos[:]

        if action == Action.UP:
            next_player_pos[1] -= 1
        elif action == Action.DOWN:
            next_player_pos[1] += 1
        elif action == Action.LEFT:
            next_player_pos[0] -= 1
        elif action == Action.RIGHT:
            next_player_pos[0] += 1
        else:
            raise ValueError()

        is_move = True
        if not (0 <= next_player_pos[0] < self.W):
            is_move = False
        elif not (0 <= next_player_pos[1] < self.H):
            is_move = False
        elif self.base_field[next_player_pos[1]][next_player_pos[0]] == 9:
            is_move = False

        if is_move:
            self.player_pos = next_player_pos

        reward = self.move_reward
        done = False

        attribute = self.base_field[self.player_pos[1]][self.player_pos[0]]
        if attribute == 1:
            reward = 1
            done = True
        elif attribute == -1:
            reward = -1
            done = True

        return tuple(self.player_pos), reward, done, {}

    # override
    def render(self, mode="human"):
        for y in range(self.H):
            s = ""
            for x in range(self.W):
                n = self.base_field[y][x]
                if self.player_pos[0] == x and self.player_pos[1] == y:  # player
                    s += "P"
                elif n == 0:  # 道
                    s += "."
                elif n == 1:  # goal
                    s += "G"
                elif n == -1:  # 穴
                    s += "X"
                else:
                    s += str(n)
            print(s)
        print("")

    # override
    def backup(self) -> Any:
        return json.dumps(self.player_pos)

    # override
    def restore(self, data: Any) -> None:
        self.player_pos = json.loads(data)

    def action_to_str(self, action) -> str:
        if Action.DOWN.value == action:
            return "↓"
        if Action.LEFT.value == action:
            return "←"
        if Action.RIGHT.value == action:
            return "→"
        if Action.UP.value == action:
            return "↑"
        return str(action)


if __name__ == "__main__":

    env = gym.make("MyEnv-v0")
    env = cast(EnvBase, env)
    env.reset()
    done = False
    total_reward = 0
    step = 0
    env.render()

    while not done:
        valid_action = env.fetch_valid_actions()
        action = random.choice(valid_action)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        step += 1
        print(f"step {step}, action {action}")
        env.render()

    print(total_reward)
