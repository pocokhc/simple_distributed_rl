import enum
from dataclasses import dataclass
from typing import Any, Tuple

from srl.base.define import EnvObservationTypes
from srl.base.env import registration
from srl.base.env.genre.singleplay import SinglePlayEnv
from srl.base.spaces.discrete import DiscreteSpace

registration.register(
    id="SampleEnv",
    entry_point=__name__ + ":SampleEnv",
    kwargs={
        "move_reward": -0.04,
    },
)


class Action(enum.Enum):
    LEFT = 0
    RIGHT = 1


@dataclass
class SampleEnv(SinglePlayEnv):
    move_reward: float = -0.04

    def __post_init__(self):
        self.field = [-1, 0, 0, 0, 0, 0, 0, 0, 1]

    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(len(Action))

    @property
    def observation_space(self) -> DiscreteSpace:
        return DiscreteSpace(len(self.field))

    @property
    def observation_type(self) -> EnvObservationTypes:
        return EnvObservationTypes.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return 20

    def call_reset(self) -> Tuple[int, dict]:
        self.player_pos = 4
        return self.player_pos, {}

    def call_step(self, action_: int) -> Tuple[int, float, bool, dict]:
        action = Action(action_)

        if action == Action.LEFT:
            self.player_pos -= 1
        elif action == Action.RIGHT:
            self.player_pos += 1

        if self.field[self.player_pos] == -1:
            return self.player_pos, -1, True, {}

        if self.field[self.player_pos] == 1:
            return self.player_pos, 1, True, {}

        return self.player_pos, self.move_reward, False, {}

    def backup(self) -> Any:
        return self.player_pos

    def restore(self, data: Any) -> None:
        self.player_pos = data

    def render_terminal(self):
        s = ""
        for x in range(len(self.field)):
            if x == self.player_pos:
                s += "P"
            elif self.field[x] == -1:
                s += "X"
            elif self.field[x] == 1:
                s += "G"
            else:
                s += "."
        print(s)

    @property
    def render_interval(self) -> float:
        return 1000 / 1
