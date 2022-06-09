import enum
import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from srl.base.define import EnvObservationType
from srl.base.env import registration
from srl.base.env.base import SpaceBase
from srl.base.env.genre import SinglePlayEnv
from srl.base.env.spaces import DiscreteSpace

logger = logging.getLogger(__name__)


registration.register(
    id="Tiger",
    entry_point=__name__ + ":Tiger",
    kwargs={},
)


class Action(enum.Enum):
    CHECK = 0
    LEFT = 1
    RIGHT = 2


class State(enum.Enum):
    WAIT = 0
    LEFT = 1
    RIGHT = 2


@dataclass
class Tiger(SinglePlayEnv):
    def __post_init__(self):
        self.prob = 0.85

    @property
    def action_space(self) -> SpaceBase:
        return DiscreteSpace(len(Action))

    @property
    def observation_space(self) -> SpaceBase:
        return DiscreteSpace(len(State))

    @property
    def observation_type(self) -> EnvObservationType:
        return EnvObservationType.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return 10

    def call_reset(self) -> np.ndarray:
        self.tiger = random.randint(0, 1)
        self.state = 0
        return np.array(self.state)

    def backup(self) -> Any:
        return json.dumps(
            [
                self.tiger,
            ]
        )

    def restore(self, data: Any) -> None:
        d = json.loads(data)
        self.tiger = d[0]

    def call_step(self, action_: int) -> Tuple[np.ndarray, float, bool, dict]:
        action = Action(action_)

        if action == Action.CHECK:
            done = False
            reward = -1
            if random.random() < self.prob:
                if self.tiger == 0:
                    self.state = 2
                else:
                    self.state = 1
            else:
                if self.tiger == 0:
                    self.state = 1
                else:
                    self.state = 2
        elif action == Action.LEFT:
            self.state = 0
            done = True
            if self.tiger == 0:
                reward = -100
            else:
                reward = 10
        elif action == Action.RIGHT:
            self.state = 0
            done = True
            if self.tiger == 0:
                reward = 10
            elif self.tiger == 1:
                reward = -100
        else:
            raise ValueError()

        return np.array(self.state), reward, done, {}

    def render_terminal(self):
        print(f"state: {self.state}, tiger: {self.tiger}")

    def action_to_str(self, action) -> str:
        if Action.CHECK.value == action:
            return "c"
        if Action.LEFT.value == action:
            return "←"
        if Action.RIGHT.value == action:
            return "→"
        return str(action)
