import json
import logging
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from srl.base.define import EnvObservationType
from srl.base.env import registration
from srl.base.env.base import SpaceBase
from srl.base.env.genre import SinglePlayEnv
from srl.base.env.spaces import BoxSpace, DiscreteSpace

logger = logging.getLogger(__name__)

registration.register(
    id="OneRoad",
    entry_point=__name__ + ":OneRoad",
    kwargs={"N": 10, "action": 2},  # 0.0009765625%
)
registration.register(
    id="OneRoad-hard",
    entry_point=__name__ + ":OneRoad",
    kwargs={"N": 20, "action": 16},
)


@dataclass
class OneRoad(SinglePlayEnv):

    N: int = 10
    action: int = 2

    @property
    def action_space(self) -> SpaceBase:
        return DiscreteSpace(self.action)

    @property
    def observation_space(self) -> SpaceBase:
        return BoxSpace(low=0, high=self.N, shape=(1,))

    @property
    def observation_type(self) -> EnvObservationType:
        return EnvObservationType.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return int(self.N * 1.1)

    def call_reset(self) -> np.ndarray:
        self.player_pos = 0
        return np.asarray(self.player_pos)

    def backup(self) -> Any:
        return json.dumps(
            [
                self.player_pos,
            ]
        )

    def restore(self, data: Any) -> None:
        d = json.loads(data)
        self.player_pos = d[0]

    def call_step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
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

        return np.asarray(self.player_pos), reward, done, {}

    def render_terminal(self):
        print(f"{self.player_pos} / {self.N}")
