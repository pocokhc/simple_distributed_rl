import enum
import logging
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import gym
import gym.envs.registration
import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType
from srl.base.env.genre.turnbase import TurnBase2PlayerActionDiscrete
from srl.base.env.registration import register
from srl.base.rl.algorithms.rulebase import RuleBaseWorker
from srl.base.rl.base import RLWorker

logger = logging.getLogger(__name__)


class StateType(enum.Enum):
    ARRAY = enum.auto()
    MAP = enum.auto()


register(
    id="OX",
    entry_point=__name__ + ":OX",
    kwargs={
        "state_type": StateType.ARRAY,
    },
)


@dataclass
class OX(TurnBase2PlayerActionDiscrete):

    state_type: StateType = StateType.ARRAY

    def __post_init__(self):

        self.W = 3
        self.H = 3

        self._player_index = 0

        # observation_space
        if self.state_type == StateType.ARRAY:
            self._observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.H * self.W,))
        elif self.state_type == StateType.MAP:
            field = self.reset()
            self._observation_space = gym.spaces.Box(low=0, high=1, shape=field.shape)
        else:
            raise ValueError()

    @property
    def action_num(self):
        return self.W * self.H

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def observation_type(self) -> EnvObservationType:
        if self.state_type == StateType.ARRAY:
            return EnvObservationType.DISCRETE
        elif self.state_type == StateType.MAP:
            return EnvObservationType.SHAPE2
        else:
            raise ValueError()

    @property
    def max_episode_steps(self) -> int:
        return 10

    @property
    def player_index(self) -> int:
        return self._player_index

    def reset_turn(self) -> Tuple[np.ndarray, np.ndarray]:
        self.field = [0 for _ in range(self.W * self.H)]
        self._player_index = 0
        return (
            self._encode_state(0, self.state_type),
            self._encode_state(1, self.state_type),
        )

    # 観測用の状態を返す
    def _encode_state(self, player, state_type):
        # 自プレイヤー：1
        # 敵プレイヤー：-1
        field = self.field[:]
        if player == 1:
            field = []
            for s in self.field:
                if s == 1:
                    field.append(-1)
                elif s == -1:
                    field.append(1)
                else:
                    field.append(0)

        if state_type == StateType.ARRAY:
            return np.array(field)
        elif state_type == StateType.MAP:
            # 0Layer: my
            # 1Layer: enemy
            _field = np.zeros((2, self.H, self.W))
            for y in range(self.H):
                for x in range(self.W):
                    idx = x + y * self.W
                    if field[idx] == 1:
                        _field[0][y][x] = 1
                    elif field[idx] == -1:
                        _field[1][y][x] = 1
            return _field

        raise ValueError()

    def backup(self) -> Any:
        return [self.field[:], self._player_index]

    def restore(self, data: Any) -> None:
        self.field = data[0][:]
        self._player_index = data[1]

    def step_turn(self, action: int) -> Tuple[np.ndarray, np.ndarray, float, float, bool, dict]:

        reward1, reward2, done = self._step(action)

        if self._player_index == 0:
            self._player_index = 1
        else:
            self._player_index = 0

        return (
            self._encode_state(0, self.state_type),
            self._encode_state(1, self.state_type),
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

    def fetch_invalid_actions_turn(self) -> Tuple[List[int], List[int]]:
        actions = []
        for a in range(self.H * self.W):
            if self.field[a] != 0:
                # x = a % self.W
                # y = a // self.W
                actions.append(a)
        return actions, actions

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

    def make_worker(self, name: str) -> Optional[RLWorker]:
        if name == "cpu_lv1":
            return NegaMax(0.5)
        elif name == "cpu_lv2":
            return NegaMax(0.1)
        elif name == "cpu_lv3":
            return NegaMax(0.0)
        return None

    def copy(self):
        env = OX(self.state_type)
        env.restore(self.backup())
        return env


class NegaMax(RuleBaseWorker):
    def __init__(self, epsilon: float):
        self.cache = {}
        self.epsilon = epsilon

    def call_on_reset(self, env) -> None:
        pass

    def call_policy(self, env_org: OX) -> int:
        env = env_org.copy()
        env.state_type = StateType.ARRAY

        if random.random() < self.epsilon:
            actions = [a for a in range(env.action_num) if env.field[a] == 0]
            return random.choice(actions)
        else:
            scores = self._negamax(env)
            return int(random.choice(np.where(scores == scores.max())[0]))

    def _negamax(self, env: OX, depth: int = 10):
        if depth <= 0:
            return 0, 0

        key = str(env.field + [env.player_index])
        if key in self.cache:
            return self.cache[key]

        scores = np.array([-9 for _ in range(env.action_num)])
        for a in range(env.action_num):
            if env.field[a] != 0:
                continue

            n_env = env.copy()
            _, _, r1, r2, done, _ = n_env.step_turn(a)
            if done:
                if env.player_index == 0:
                    scores[a] = r1
                else:
                    scores[a] = r2
            else:
                n_scores = self._negamax(n_env, depth - 1)
                scores[a] = -np.max(n_scores)

        self.cache[key] = scores

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


if __name__ == "__main__":
    pass
