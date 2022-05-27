import logging
import random
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from srl.base.define import EnvObservationType
from srl.base.env.base import SpaceBase
from srl.base.env.genre import TurnBase2Player
from srl.base.env.registration import register
from srl.base.env.spaces import BoxSpace, DiscreteSpace
from srl.base.rl.algorithms.rulebase import RuleBaseWorker
from srl.base.rl.base import RLWorker

logger = logging.getLogger(__name__)

register(
    id="StoneTaking",
    entry_point=__name__ + ":StoneTaking",
    kwargs={},
)


@dataclass
class StoneTaking(TurnBase2Player):

    stones: int = 10
    max_stones: int = 3

    def __post_init__(self):
        pass  #

    @property
    def action_space(self) -> SpaceBase:
        return DiscreteSpace(self.max_stones)

    @property
    def observation_space(self) -> SpaceBase:
        return BoxSpace(
            low=0,
            high=self.stones,
            shape=(1,),
        )

    @property
    def observation_type(self) -> EnvObservationType:
        return EnvObservationType.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return self.stones

    @property
    def player_index(self) -> int:
        return self._player_index

    def call_reset(self) -> np.ndarray:
        self.field = self.stones
        self._player_index = 0
        return np.array([self.field])

    def backup(self) -> Any:
        return [self.field, self._player_index]

    def restore(self, data: Any) -> None:
        self.field = data[0]
        self._player_index = data[1]

    def call_step(self, action: int) -> Tuple[np.ndarray, float, float, bool, dict]:
        action += 1

        reward1, reward2, done = self._step(action)

        if self._player_index == 0:
            self._player_index = 1
        else:
            self._player_index = 0

        return (
            np.array([self.field]),
            reward1,
            reward2,
            done,
            {},
        )

    def _step(self, action):

        self.field -= action
        if self.field > 0:
            return 0, 0, False
        self.field = 0

        # 最後の石を取ったら負け
        if self.player_index == 0:
            return -1, 1, True
        else:
            return 1, -1, True

    def render_terminal(self):
        s = ""
        for _ in range(self.field):
            s += "o"
        print(f"{self.field:3d}: {s}")
        print(f"next player: {self.player_index}")

    def action_to_str(self, action: int) -> str:
        return str(action + 1)

    def make_worker(self, name: str) -> Optional[RLWorker]:
        if name == "cpu":
            return CPU()
        return None


class CPU(RuleBaseWorker):
    def __init__(self):
        pass  #

    def call_on_reset(self, env) -> None:
        pass  #

    def call_policy(self, env: StoneTaking) -> int:
        if env.field in [1, 2, 6, 10, 14, 18]:
            return 1 - 1
        if env.field in [3, 7, 11, 15, 19]:
            return 2 - 1
        if env.field in [4, 8, 12, 16, 20]:
            return 3 - 1
        return random.randint(0, 2)
