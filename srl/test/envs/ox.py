import logging
from typing import Any, List, Tuple, cast

import numpy as np
from srl.base.define import EnvObservationType, RLObservationType
from srl.base.env.base import EnvRun, SpaceBase
from srl.base.env.genre import TurnBase2Player
from srl.base.env.registration import register
from srl.base.env.spaces import ArrayDiscreteSpace, BoxSpace, DiscreteSpace
from srl.base.rl.processor import Processor

logger = logging.getLogger(__name__)

register(
    id="TestOX",
    entry_point=__name__ + ":OX",
    kwargs={},
)


class OX(TurnBase2Player):
    _scores_cache = {}

    def __init__(self):

        self.W = 3
        self.H = 3

        self._player_index = 0

    @property
    def action_space(self) -> SpaceBase:
        return DiscreteSpace(self.W * self.H)

    @property
    def observation_space(self) -> SpaceBase:
        return ArrayDiscreteSpace(self.H * self.W, low=-1, high=1)

    @property
    def observation_type(self) -> EnvObservationType:
        return EnvObservationType.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return 10

    @property
    def player_index(self) -> int:
        return self._player_index

    def call_reset(self) -> Tuple[List[int], dict]:
        self.field = [0 for _ in range(self.W * self.H)]
        self._player_index = 0
        return self.field, {}

    def backup(self) -> Any:
        return [self.field[:], self._player_index]

    def restore(self, data: Any) -> None:
        self.field = data[0][:]
        self._player_index = data[1]

    def call_step(self, action: int) -> Tuple[List[int], float, float, bool, dict]:

        reward1, reward2, done = self._step(action)

        if not done:
            if self._player_index == 0:
                self._player_index = 1
            else:
                self._player_index = 0

        return self.field, reward1, reward2, done, {}

    def _step(self, action):

        # error action
        if self.field[action] != 0:
            if self.player_index == 0:
                return -1, 0, True
            else:
                return 0, -1, True

        # update
        if self.player_index == 0:
            self.field[action] = 1
        else:
            self.field[action] = -1

        reward1, reward2, done = self._check(self.field)

        return reward1, reward2, done

    def _check(self, field):
        for pos in [
            # 横
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            # 縦
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            # 斜め
            [0, 4, 8],
            [2, 4, 6],
        ]:
            # 3つとも同じか
            if field[pos[0]] == field[pos[1]] and field[pos[1]] == field[pos[2]]:
                if field[pos[0]] == 1:
                    # player1 win
                    return 1, -1, True
                if field[pos[0]] == -1:
                    # player2 win
                    return -1, 1, True

        # 置く場所がなければdraw
        if sum([1 if v == 0 else 0 for v in field]) == 0:
            return 0, 0, True

        return 0, 0, False

    def get_invalid_actions(self, player_index: int = 0) -> List[int]:
        actions = []
        for a in range(self.H * self.W):
            if self.field[a] != 0:
                # x = a % self.W
                # y = a // self.W
                actions.append(a)
        return actions

    def render_terminal(self, **kwargs) -> None:
        print("-" * 10)
        for y in range(self.H):
            s = "|"
            for x in range(self.W):
                a = x + y * self.W
                if self.field[a] == 1:
                    s += " o|"
                elif self.field[a] == -1:
                    s += " x|"
                else:
                    s += "{:2d}|".format(a)
            print(s)
            print("-" * 10)
        if self.player_index == 0:
            print("next player: O")
        else:
            print("next player: X")


class LayerProcessor(Processor):
    def change_observation_info(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
        rl_observation_type: RLObservationType,
        env: OX,
    ) -> Tuple[SpaceBase, EnvObservationType]:
        observation_space = BoxSpace(
            low=0,
            high=1,
            shape=(2, 3, 3),
        )
        return observation_space, EnvObservationType.SHAPE3

    def process_observation(self, observation: np.ndarray, _env: EnvRun) -> np.ndarray:
        env = cast(OX, _env.get_original_env())

        # Layer0: player1 field (0 or 1)
        # Layer1: player2 field (0 or 1)
        if env.player_index == 0:
            my_field = 1
            enemy_field = -1
        else:
            my_field = -1
            enemy_field = 1
        _field = np.zeros((2, env.H, env.W))
        for y in range(env.H):
            for x in range(env.W):
                idx = x + y * env.W
                if observation[idx] == my_field:
                    _field[0][y][x] = 1
                elif observation[idx] == enemy_field:
                    _field[1][y][x] = 1
        return _field
