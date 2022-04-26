import enum
import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Tuple

import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType
from srl.base.env import registration
from srl.base.env.genre.singleplay import SingleActionDiscrete

logger = logging.getLogger(__name__)


registration.register(
    id="IGrid",
    entry_point=__name__ + ":IGrid",
    kwargs={
        "N": 0,
    },
)


class Action(enum.Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


@dataclass
class IGrid(SingleActionDiscrete):
    """
    CXD
     X
     S × N
     X
    AXB

    X: 通路、真ん中の通路数は可変（N*2+1個）
    S: 開始地点
    A,B: key
    C,D: GOAL
    Aを取った場合はCがゴール(+1)になり、Dが落とし穴(-1)になる
    Bを取った場合はCが落とし穴(-1)になり、Dがゴール(+1)になる
    """

    N: int = 0

    def __post_init__(self):
        self.length = self.N * 2 + 1

    @property
    def action_num(self) -> int:
        return len(Action)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(
            low=0,
            high=np.maximum(self.H, self.W),
            shape=(2,),
        )

    @property
    def observation_type(self) -> EnvObservationType:
        return EnvObservationType.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return (self.length + 2) * 2 * 2

    def reset_single(self) -> Any:
        self.player_pos = (1, int((self.length + 2 - 1) / 2))

        self.field = [[1, 1, 1]]
        for _ in range(self.length):
            self.field.append([0, 1, 0])
        self.field.append([2, 1, 3])

        return np.array(self.player_pos)

    def backup(self) -> Any:
        return json.dumps(
            [
                self.player_pos,
                self.field,
            ]
        )

    def restore(self, data: Any) -> None:
        d = json.loads(data)
        self.player_pos = d[0]
        self.field = d[1]

    def step_single(self, action_: int) -> Tuple[Any, float, bool, dict]:
        action = Action(action_)

        x = self.player_pos[0]
        y = self.player_pos[1]

        if action == Action.UP:
            y -= 1
        elif action == Action.DOWN:
            y += 1
        elif action == Action.LEFT:
            x -= 1
        elif action == Action.RIGHT:
            x += 1
        else:
            raise ValueError()

        # check
        is_move = True
        if x < 0 or x >= self.W:
            is_move = False
        elif y < 0 or y >= self.H:
            is_move = False
        elif self.field[y][x] == 0:
            is_move = False
        if is_move:
            self.player_pos = (x, y)
        x = self.player_pos[0]
        y = self.player_pos[1]

        reward = 0
        done = False
        if self.field[y][x] == 2:
            self.field[0][0] = 4
            self.field[0][2] = 5
        elif self.field[y][x] == 3:
            self.field[0][0] = 5
            self.field[0][2] = 4
        elif self.field[y][x] == 4:
            reward = 1
            done = True
        elif self.field[y][x] == 5:
            reward = -1
            done = True

        return np.array(self.player_pos), reward, done, {}

    def render_terminal(self):
        for y in range(self.H):
            s = ""
            for x in range(self.W):
                if self.player_pos == (x, y):
                    s += "P"
                elif self.field[y][x] == 0:
                    s += " "
                elif self.field[y][x] == 1:
                    s += "."
                elif self.field[y][x] == 2:
                    s += "K"
                elif self.field[y][x] == 3:
                    s += "K"
                elif self.field[y][x] == 4:
                    s += "G"
                elif self.field[y][x] == 5:
                    s += "X"
                else:
                    s += " "
            print(s)
        print("")

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

    # ------------------------------------
    @property
    def W(self) -> int:
        return 3

    @property
    def H(self) -> int:
        return self.length + 2

    @property
    def actions(self):
        return [a for a in Action]


if __name__ == "__main__":
    game = IGrid()
    game.reset()
    done = False
    total_reward = 0
    step = 0
    game.render()

    while not done:
        action = game.sample()
        state, reward, done, _ = game.step(action)
        total_reward += reward
        step += 1
        print(f"step {step}, action {action}")
        game.render()

    print(total_reward)
