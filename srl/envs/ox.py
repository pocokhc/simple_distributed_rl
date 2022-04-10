import logging
import random
from dataclasses import dataclass
from typing import Any, List, Tuple

import gym
import gym.envs.registration
import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType
from srl.base.env import EnvBase

logger = logging.getLogger(__name__)


gym.envs.registration.register(
    id="OX-v0",
    entry_point=__name__ + ":OX",
    kwargs={},
)
gym.envs.registration.register(
    id="OX2d-v0",
    entry_point=__name__ + ":OX",
    kwargs={
        "enable_state_encode": True,
    },
)


@dataclass
class OX(EnvBase):

    enable_state_encode: bool = False

    def __post_init__(self):

        self.W = 3
        self.H = 3

        self._action_space = gym.spaces.Discrete(self.W * self.H)

        # observation_space
        if self.enable_state_encode:
            state = self.reset()
            self._observation_space = gym.spaces.Box(low=0, high=1, shape=state.shape)
        else:
            self._observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.H * self.W,))

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def observation_type(self) -> EnvObservationType:
        if self.enable_state_encode:
            return EnvObservationType.SHAPE2
        else:
            return EnvObservationType.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return 5

    def reset(self) -> Any:
        self.state = [0 for _ in range(self.W * self.H)]
        if random.randint(0, 1) == 1:
            valid_actions = self.fetch_valid_actions()
            action = random.choice(valid_actions)
            self.state, reward, done = self._step(self.state, -1, action)
        return self._encode_state()

    # --- 状態設計
    # o: 1Layer
    # ×: 1Layer
    def _encode_state(self):
        if not self.enable_state_encode:
            return np.asarray(self.state)
        field = np.zeros((2, self.H, self.W))
        for y in range(self.H):
            for x in range(self.W):
                idx = x + y * self.W
                if self.state[idx] == 1:
                    field[0][y][x] = 1
                elif self.state[idx] == -1:
                    field[1][y][x] = 1
        return field

    def step(self, action: int) -> Tuple[Any, float, bool, dict]:

        # 自分のターン
        self.state, reward, done = self._step(self.state, 1, action)
        if done:
            return self._encode_state(), reward, done, {}

        # 相手のターン
        valid_actions = self.fetch_valid_actions()
        action = random.choice(valid_actions)
        self.state, reward, done = self._step(self.state, -1, action)
        return self._encode_state(), reward, done, {}

    def _step(self, state, player, action):

        # deepcopy
        n_state = state[:]
        if n_state[action] != 0:
            # assert False
            return n_state, -1, True

        n_state[action] = player

        # --- チェック
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
            n = n_state[pos[0]] + n_state[pos[1]] + n_state[pos[2]]
            if n == 3:
                # player 1 win
                return n_state, 1, True
            if n == -3:
                # player 2 win
                return n_state, -1, True

        # 0の数
        n = sum([1 if v == 0 else 0 for v in n_state])
        if n == 0:
            # draw
            return n_state, 0, True

        return n_state, 0, False

    def fetch_valid_actions(self) -> List[int]:
        state = self.state

        actions = []
        for a in range(self.H * self.W):
            if state[a] == 0:
                # x = a % self.W
                # y = a // self.W
                actions.append(a)
        return actions

    def render(self, mode: str = "human") -> Any:  # super
        state = self.state
        valid_actions = self.fetch_valid_actions()

        print("-" * 10)
        for y in range(self.H):
            s = "|"
            for x in range(self.W):
                a = x + y * self.W
                if state[a] == 1:
                    s += " o|"
                elif state[a] == -1:
                    s += " x|"
                else:
                    is_act = False
                    for va in valid_actions:
                        if va == a:
                            s += "{:2d}|".format(a)
                            is_act = True
                            break
                    if not is_act:
                        s += "  |"
            print(s)
            print("-" * 10)

    def backup(self) -> Any:
        return self.state[:]

    def restore(self, state: Any) -> None:
        self.state = state[:]


if __name__ == "__main__":
    pass
