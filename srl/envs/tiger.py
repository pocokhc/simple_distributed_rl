import enum
import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from srl.base.env import registration
from srl.base.env.base import EnvBase
from srl.base.spaces.discrete import DiscreteSpace

logger = logging.getLogger(__name__)


registration.register(
    id="Tiger",
    entry_point=__name__ + ":Tiger",
    kwargs={},
    check_duplicate=False,
)


class Action(enum.Enum):
    CHECK = 0
    LEFT = 1
    RIGHT = 2


class State(enum.Enum):
    LEFT = 0
    RIGHT = 1


@dataclass
class Tiger(EnvBase[DiscreteSpace, int, DiscreteSpace, int]):
    def __post_init__(self):
        self.prob = 0.85

    @property
    def action_space(self):
        return DiscreteSpace(len(Action))

    @property
    def observation_space(self):
        return DiscreteSpace(len(State))

    @property
    def player_num(self) -> int:
        return 1

    @property
    def max_episode_steps(self) -> int:
        return 10

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> Any:
        self.tiger = State(random.randint(0, 1))
        self.state = State(random.randint(0, 1))
        return self.state.value

    def backup(self) -> Any:
        return json.dumps(
            [
                self.tiger.value,
                self.state.value,
            ]
        )

    def restore(self, data: Any) -> None:
        d = json.loads(data)
        self.tiger = State(d[0])
        self.state = State(d[1])

    def step(self, action) -> Tuple[int, float, bool, bool]:
        action = Action(action)

        if action == Action.CHECK:
            done = False
            reward = -1
            if random.random() < self.prob:
                if self.tiger == State.LEFT:
                    self.state = State.LEFT
                else:
                    self.state = State.RIGHT
            else:
                if self.tiger == State.RIGHT:
                    self.state = State.LEFT
                else:
                    self.state = State.RIGHT
        elif action == Action.LEFT:
            done = True
            self.state = self.tiger
            if self.tiger == State.LEFT:
                reward = -100
            else:
                reward = 10
        elif action == Action.RIGHT:
            done = True
            self.state = self.tiger
            if self.tiger == State.LEFT:
                reward = 10
            else:
                reward = -100
        else:
            raise ValueError()

        return self.state.value, float(reward), done, False

    def render_terminal(self):
        print(f"state: {self.state}, tiger: {self.tiger}")

    @property
    def render_interval(self) -> float:
        return 1000 / 1

    def action_to_str(self, action) -> str:
        if Action.CHECK.value == action:
            return "c"
        if Action.LEFT.value == action:
            return "←"
        if Action.RIGHT.value == action:
            return "→"
        return str(action)
