import logging
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

from srl.base.define import EnvActionType
from srl.base.env.base import EnvBase
from srl.base.env.env_run import EnvRun
from srl.base.env.registration import register
from srl.base.rl.algorithms.env_worker import EnvWorker
from srl.base.spaces.discrete import DiscreteSpace

logger = logging.getLogger(__name__)

register(
    id="StoneTaking",
    entry_point=__name__ + ":StoneTaking",
    kwargs={},
    check_duplicate=False,
)


@dataclass
class StoneTaking(EnvBase[DiscreteSpace, int, DiscreteSpace, int]):
    stones: int = 10
    max_stones: int = 3

    def __post_init__(self):
        pass  #

    @property
    def action_space(self):
        return DiscreteSpace(self.max_stones)

    @property
    def observation_space(self):
        return DiscreteSpace(self.stones + 1)

    @property
    def player_num(self) -> int:
        return 2

    @property
    def max_episode_steps(self) -> int:
        return self.stones

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> Any:
        self.field = self.stones
        self.next_player = 0
        return self.field

    def backup(self) -> Any:
        return self.field

    def restore(self, data: Any) -> None:
        self.field = data

    def step(self, action) -> Tuple[int, List[float], bool, bool]:
        action += 1

        reward1, reward2, done = self._step(action)

        if self.next_player == 0:
            self.next_player = 1
        else:
            self.next_player = 0

        return self.field, [reward1, reward2], done, False

    def _step(self, action):
        self.field -= action
        if self.field > 0:
            return 0.0, 0.0, False
        self.field = 0

        # 最後の石を取ったら負け
        if self.next_player == 0:
            return -1.0, 1.0, True
        else:
            return 1.0, -1.0, True

    def render_terminal(self):
        s = ""
        for _ in range(self.field):
            s += "o"
        print(f"{self.field:3d}: {s}")
        print(f"next player: {self.next_player}")

    @property
    def render_interval(self) -> float:
        return 1000 / 1

    def action_to_str(self, action: int) -> str:
        return str(action + 1)

    def make_worker(self, name: str, **kwargs) -> Optional[EnvWorker]:
        if name == "cpu":
            return CPU(**kwargs)
        return None


class CPU(EnvWorker):
    def call_policy(self, env: EnvRun) -> Tuple[EnvActionType, dict]:
        _env = cast(StoneTaking, env.unwrapped)
        if _env.field == 1:
            return 0, {}

        if _env.field % 4 == 2:
            return 1 - 1, {}
        if _env.field % 4 == 3:
            return 2 - 1, {}
        if _env.field % 4 == 0:
            return 3 - 1, {}

        return random.randint(0, 2), {}
