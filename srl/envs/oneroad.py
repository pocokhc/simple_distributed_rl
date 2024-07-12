import json
import logging
from dataclasses import dataclass
from typing import Any, Tuple

from srl.base.env import registration
from srl.base.env.genre import SinglePlayEnv
from srl.base.spaces import DiscreteSpace

logger = logging.getLogger(__name__)

registration.register(
    id="OneRoad",
    entry_point=__name__ + ":OneRoad",
    kwargs={"N": 10, "action": 2, "is_end": True},
    check_duplicate=False,
)
registration.register(
    id="OneRoad-hard",
    entry_point=__name__ + ":OneRoad",
    kwargs={"N": 20, "action": 16, "is_end": False},
    check_duplicate=False,
)


@dataclass
class OneRoad(SinglePlayEnv):
    N: int = 10
    action: int = 2
    is_end: bool = True

    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(self.action)

    @property
    def observation_space(self) -> DiscreteSpace:
        return DiscreteSpace(self.N)

    @property
    def max_episode_steps(self) -> int:
        return int(self.N * 1.1)

    def call_reset(self) -> Tuple[int, dict]:
        self.player_pos = 0
        return self.player_pos, {}

    def backup(self) -> Any:
        return json.dumps(
            [
                self.player_pos,
            ]
        )

    def restore(self, data: Any) -> None:
        d = json.loads(data)
        self.player_pos = d[0]

    def call_step(self, action: int) -> Tuple[int, float, bool, dict]:
        if action == 0:
            self.player_pos += 1
        else:
            if self.is_end:
                return self.player_pos, 0.0, True, {}
            else:
                self.player_pos = 0

        if self.player_pos == self.N:
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False

        return self.player_pos, reward, done, {}

    def render_terminal(self):
        print(f"{self.player_pos} / {self.N}")

    @property
    def render_interval(self) -> float:
        return 1000 / 1
