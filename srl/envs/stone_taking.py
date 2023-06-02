import logging
import random
from dataclasses import dataclass
from typing import Any, Optional, Tuple, cast

from srl.base.define import EnvActionType, EnvObservationTypes
from srl.base.env.env_run import EnvRun
from srl.base.env.genre import TurnBase2Player
from srl.base.env.registration import register
from srl.base.rl.worker import RuleBaseWorker, WorkerRun
from srl.base.spaces import DiscreteSpace

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
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(self.max_stones)

    @property
    def observation_space(self) -> DiscreteSpace:
        return DiscreteSpace(self.stones + 1)

    @property
    def observation_type(self) -> EnvObservationTypes:
        return EnvObservationTypes.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return self.stones

    @property
    def next_player_index(self) -> int:
        return self._next_player_index

    def call_reset(self) -> Tuple[int, dict]:
        self.field = self.stones
        self._next_player_index = 0
        return self.field, {}

    def backup(self) -> Any:
        return [self.field, self._next_player_index]

    def restore(self, data: Any) -> None:
        self.field = data[0]
        self._next_player_index = data[1]

    def call_step(self, action: int) -> Tuple[int, float, float, bool, dict]:
        action += 1

        reward1, reward2, done = self._step(action)

        if self._next_player_index == 0:
            self._next_player_index = 1
        else:
            self._next_player_index = 0

        return self.field, reward1, reward2, done, {}

    def _step(self, action):
        self.field -= action
        if self.field > 0:
            return 0, 0, False
        self.field = 0

        # 最後の石を取ったら負け
        if self._next_player_index == 0:
            return -1, 1, True
        else:
            return 1, -1, True

    def render_terminal(self):
        s = ""
        for _ in range(self.field):
            s += "o"
        print(f"{self.field:3d}: {s}")
        print(f"next player: {self._next_player_index}")

    @property
    def render_interval(self) -> float:
        return 1000 / 1

    def action_to_str(self, action: int) -> str:
        return str(action + 1)

    def make_worker(self, name: str, **kwargs) -> Optional[RuleBaseWorker]:
        if name == "cpu":
            return CPU(**kwargs)
        return None


class CPU(RuleBaseWorker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call_on_reset(self, env: EnvRun, worker: WorkerRun) -> dict:
        return {}

    def call_policy(self, env: EnvRun, worker: WorkerRun) -> Tuple[EnvActionType, dict]:
        _env = cast(StoneTaking, env.get_original_env())
        if _env.field == 1:
            return 0, {}

        if _env.field % 4 == 2:
            return 1 - 1, {}
        if _env.field % 4 == 3:
            return 2 - 1, {}
        if _env.field % 4 == 0:
            return 3 - 1, {}

        return random.randint(0, 2), {}
