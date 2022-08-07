import logging
import random
from dataclasses import dataclass
from typing import Any, Optional, Tuple, cast

import numpy as np
from srl.base.define import EnvAction, EnvObservationType
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.env.genre import TurnBase2Player
from srl.base.env.registration import register
from srl.base.env.spaces import BoxSpace, DiscreteSpace
from srl.base.rl.base import RuleBaseWorker, WorkerRun

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

    def make_worker(self, name: str) -> Optional[RuleBaseWorker]:
        if name == "cpu":
            return CPU()
        return None


class CPU(RuleBaseWorker):
    def call_on_reset(self, env: EnvRun, worker: WorkerRun) -> None:
        pass  #

    def call_policy(self, _env: EnvRun, worker: WorkerRun) -> EnvAction:
        env = cast(StoneTaking, _env.get_original_env())
        if env.field == 1:
            return 0

        if env.field % 4 == 2:
            return 1 - 1
        if env.field % 4 == 3:
            return 2 - 1
        if env.field % 4 == 0:
            return 3 - 1

        return random.randint(0, 2)

    def call_render(self, _env: EnvRun, worker_run: WorkerRun) -> None:
        pass  #
