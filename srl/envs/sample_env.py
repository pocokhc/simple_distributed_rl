import enum
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from srl.base.define import KeyBindType
from srl.base.env import registration
from srl.base.env.base import EnvBase
from srl.base.spaces.discrete import DiscreteSpace

registration.register(
    id="SampleEnv",
    entry_point=__name__ + ":SampleEnv",
    kwargs={
        "move_reward": -0.04,
    },
    check_duplicate=False,
)


class Action(enum.Enum):
    LEFT = 0
    RIGHT = 1


@dataclass
class SampleEnv(EnvBase[DiscreteSpace, int, DiscreteSpace, int]):
    move_reward: float = -0.04

    def __post_init__(self):
        super().__init__()
        self.field = [-1, 0, 0, 0, 0, 0, 0, 0, 1]

    @property
    def action_space(self):
        return DiscreteSpace(len(Action))

    @property
    def observation_space(self):
        return DiscreteSpace(len(self.field))

    @property
    def player_num(self) -> int:
        return 1

    @property
    def max_episode_steps(self) -> int:
        return 20

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> Any:
        self.player_pos = 4
        return self.player_pos

    def step(self, action) -> Tuple[int, float, bool, bool]:
        action = Action(action)

        if action == Action.LEFT:
            self.player_pos -= 1
        elif action == Action.RIGHT:
            self.player_pos += 1

        if self.field[self.player_pos] == -1:
            return self.player_pos, -1.0, True, False

        if self.field[self.player_pos] == 1:
            return self.player_pos, 1.0, True, False

        return self.player_pos, self.move_reward, False, False

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

    def action_to_str(self, action) -> str:
        if Action.LEFT.value == action:
            return "←"
        if Action.RIGHT.value == action:
            return "→"
        return str(action)

    def get_key_bind(self) -> Optional[KeyBindType]:
        return {
            "": Action.LEFT.value,
            "a": Action.LEFT.value,
            "d": Action.RIGHT.value,
        }

    @property
    def render_interval(self) -> float:
        return 1000 / 1
