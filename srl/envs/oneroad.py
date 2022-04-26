import json
import logging
from dataclasses import dataclass
from typing import Any, List, Tuple

import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType
from srl.base.env import registration
from srl.base.env.genre.singleplay import SingleActionDiscrete

logger = logging.getLogger(__name__)

registration.register(
    id="OneRoad",
    entry_point=__name__ + ":OneRoad",
    kwargs={"N": 10, "action": 2},  # 0.0009765625%
)
registration.register(
    id="OneRoad-hard",
    entry_point=__name__ + ":OneRoad",
    kwargs={"N": 30, "action": 16},
)


@dataclass
class OneRoad(SingleActionDiscrete):

    N: int = 10
    action: int = 2

    @property
    def action_num(self) -> int:
        return self.action

    @property
    def observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=0, high=self.N, shape=(1,))

    @property
    def observation_type(self) -> EnvObservationType:
        return EnvObservationType.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return int(self.N * 1.1)

    def reset_single(self) -> np.ndarray:
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

    def step_single(self, action: int) -> Tuple[Any, float, bool, dict]:
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


if __name__ == "__main__":
    pass
