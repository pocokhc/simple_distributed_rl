import enum
import json
import logging
from dataclasses import dataclass
from typing import Any, List, Tuple

from srl.base.define import EnvObservationTypes
from srl.base.env import registration
from srl.base.env.genre import SinglePlayEnv
from srl.base.env.spaces import DiscreteSpace
from srl.base.env.spaces.array_discrete import ArrayDiscreteSpace

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
class IGrid(SinglePlayEnv):
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
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(len(Action))

    @property
    def observation_space(self) -> ArrayDiscreteSpace:
        return ArrayDiscreteSpace(2, 0, [self.W, self.H])

    @property
    def observation_type(self) -> EnvObservationTypes:
        return EnvObservationTypes.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return (self.length + 2) * 2 * 2

    def call_reset(self) -> Tuple[List[int], dict]:
        self.player_pos = (1, int((self.length + 2 - 1) / 2))

        self.field = [[1, 1, 1]]
        for _ in range(self.length):
            self.field.append([0, 1, 0])
        self.field.append([2, 1, 3])

        return list(self.player_pos), {}

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

    def call_step(self, action_: int) -> Tuple[List[int], float, bool, dict]:
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

        return list(self.player_pos), reward, done, {}

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

    @property
    def render_interval(self) -> float:
        return 1000 / 1

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
