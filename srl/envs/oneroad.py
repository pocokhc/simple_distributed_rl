import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from srl.base.env import registration
from srl.base.env.base import EnvBase
from srl.base.spaces.discrete import DiscreteSpace

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
class OneRoad(EnvBase[DiscreteSpace, int, DiscreteSpace, int]):
    N: int = 10
    action: int = 2
    is_end: bool = True

    @property
    def action_space(self):
        return DiscreteSpace(self.action)

    @property
    def observation_space(self):
        return DiscreteSpace(self.N)

    @property
    def player_num(self) -> int:
        return 1

    @property
    def max_episode_steps(self) -> int:
        return int(self.N * 1.1)

    @property
    def reward_baseline(self) -> dict:
        # 乱数要素なし
        return {"episode": 100, "baseline": 1.0}

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> int:
        self.player_pos = 0
        return self.player_pos

    def backup(self) -> Any:
        return self.player_pos

    def restore(self, data: Any) -> None:
        self.player_pos = data

    def step(self, action: int) -> Tuple[int, float, bool, bool]:
        if action == 0:
            self.player_pos += 1
        else:
            if self.is_end:
                return self.player_pos, 0.0, True, False
            else:
                self.player_pos = 0

        if self.player_pos == self.N:
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False

        return self.player_pos, reward, done, False

    def render_terminal(self):
        print(f"{self.player_pos} / {self.N}")

    @property
    def render_interval(self) -> float:
        return 1000 / 1
