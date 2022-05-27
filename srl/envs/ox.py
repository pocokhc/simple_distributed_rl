import logging
import random
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
from srl.base.define import EnvObservationType, RLObservationType
from srl.base.env.base import SpaceBase
from srl.base.env.genre import TurnBase2Player
from srl.base.env.registration import register
from srl.base.env.spaces import BoxSpace, DiscreteSpace
from srl.base.rl.algorithms.rulebase import RuleBaseWorker
from srl.base.rl.processor import Processor

logger = logging.getLogger(__name__)

register(
    id="OX",
    entry_point=__name__ + ":OX",
    kwargs={},
)


@dataclass
class OX(TurnBase2Player):
    def __post_init__(self):

        self.W = 3
        self.H = 3

        self._player_index = 0

    @property
    def action_space(self) -> SpaceBase:
        return DiscreteSpace(self.W * self.H)

    @property
    def observation_space(self) -> SpaceBase:
        return BoxSpace(
            low=-1,
            high=1,
            shape=(self.H * self.W,),
        )

    @property
    def observation_type(self) -> EnvObservationType:
        return EnvObservationType.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return 10

    @property
    def player_index(self) -> int:
        return self._player_index

    def call_reset(self) -> np.ndarray:
        self.field = [0 for _ in range(self.W * self.H)]
        self._player_index = 0
        return self._encode_state()

    # 観測用の状態を返す
    def _encode_state(self):
        # (turn,) + field
        return np.array(self.field)

    def backup(self) -> Any:
        return [self.field[:], self._player_index]

    def restore(self, data: Any) -> None:
        self.field = data[0][:]
        self._player_index = data[1]

    def call_step(self, action: int) -> Tuple[np.ndarray, float, float, bool, dict]:

        reward1, reward2, done = self._step(action)

        if self._player_index == 0:
            self._player_index = 1
        else:
            self._player_index = 0

        return (
            self._encode_state(),
            reward1,
            reward2,
            done,
            {},
        )

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

    def get_invalid_actions(self, player_index: int) -> List[int]:
        actions = []
        for a in range(self.H * self.W):
            if self.field[a] != 0:
                # x = a % self.W
                # y = a // self.W
                actions.append(a)
        return actions

    def render_terminal(self):
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
        print(f"next player: {self.player_index}")

    def make_worker(self, name: str):
        if name == "cpu":
            return NegaMax
        return None


class NegaMax(RuleBaseWorker):
    cache = {}

    def __init__(self, *args):
        super().__init__(*args)

    def call_on_reset(self, env: OX) -> None:
        self.epsilon = 0.0

    def call_policy(self, env_org: OX) -> int:
        env = env_org.copy()

        if random.random() < self.epsilon:
            actions = [a for a in range(env.action_space.n) if env.field[a] == 0]
            return random.choice(actions)
        else:
            scores = self._negamax(env)
            return int(random.choice(np.where(scores == scores.max())[0]))

    def _negamax(self, env: OX, depth: int = 10):
        if depth <= 0:
            return 0, 0

        key = str(env.field + [env.player_index])
        if key in NegaMax.cache:
            return NegaMax.cache[key]

        scores = np.array([-9 for _ in range(env.action_space.n)])
        for a in range(env.action_space.n):
            if env.field[a] != 0:
                continue

            n_env = env.copy()
            _, r1, r2, done, _ = n_env.call_step(a)
            if done:
                if env.player_index == 0:
                    scores[a] = r1
                else:
                    scores[a] = r2
            else:
                n_scores = self._negamax(n_env, depth - 1)
                scores[a] = -np.max(n_scores)

        NegaMax.cache[key] = scores

        return scores

    def call_render(self, env: OX) -> None:
        scores = self._negamax(env)

        print("- negamax -")
        print("-" * 10)
        for y in range(env.H):
            s = "|"
            for x in range(env.W):
                a = x + y * env.W
                s += "{:2d}|".format(scores[a])
            print(s)
            print("-" * 10)


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
            shape=(3, 3, 3),
        )
        return observation_space, EnvObservationType.SHAPE3

    def process_observation(self, observation: np.ndarray, env: OX) -> np.ndarray:
        # Layer0: player1 field (0 or 1)
        # Layer1: player2 field (0 or 1)
        # Layer2: player_index (all0 or all1)
        _field = np.zeros((3, env.H, env.W))
        for y in range(env.H):
            for x in range(env.W):
                idx = x + y * env.W
                if env.field[idx] == 1:
                    _field[0][y][x] = 1
                elif env.field[idx] == -1:
                    _field[1][y][x] = 1
        _field[2] = env.player_index
        return _field
