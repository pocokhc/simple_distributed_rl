import json
import random
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from srl.base.env.base import EnvBase
from srl.base.rl.algorithms.table import TableConfig
from srl.base.rl.base import RLParameter, RLTrainer, RLWorker
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import SequenceRemoteMemory
from srl.rl.functions.common import to_str_observation


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(TableConfig):

    simulation_times: int = 10
    action_select_threshold: int = 5
    gamma: float = 1.0  # 割引率
    uct_c: float = np.sqrt(2.0)

    def __post_init__(self):
        super().__init__()

    @staticmethod
    def getName() -> str:
        return "MCTS"


register(
    Config,
    __name__ + ":RemoteMemory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


# ------------------------------------------------------
# RemoteMemory
# ------------------------------------------------------
class RemoteMemory(SequenceRemoteMemory):
    pass


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.N = {}  # 訪問回数
        self.W = {}  # 累計報酬

    def restore(self, data: Any) -> None:
        d = json.loads(data)
        self.N = d[0]
        self.W = d[1]

    def backup(self):
        return json.dumps(
            [
                self.N,
                self.W,
            ]
        )

    # ------------------------

    def init_state(self, state):
        if state not in self.N:
            self.W[state] = [0 for _ in range(self.config.nb_actions)]
            self.N[state] = [0 for _ in range(self.config.nb_actions)]


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        self.train_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):
        batchs = self.remote_memory.sample()
        for batch in batchs:
            state = batch["state"]
            action = batch["action"]
            reward = batch["reward"]

            self.parameter.N[state][action] += 1
            self.parameter.W[state][action] += reward
            self.train_count += 1
        return {}


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

    def on_reset(self, state: np.ndarray, player_index: int, env: EnvBase) -> None:
        self.state = to_str_observation(state)
        self.invalid_actions = env.get_invalid_actions(player_index)

    def policy(self, _state: np.ndarray, player_index: int, env: EnvBase) -> int:
        state = to_str_observation(_state)

        if self.training:
            dat = env.backup()
            for _ in range(self.config.simulation_times):
                self._simulation(env, state, player_index)
                env.restore(dat)

        # 試行回数のもっとも多いアクションを採用
        if state in self.parameter.N:
            c = self.parameter.N[state]
            c = [-np.inf if a in self.invalid_actions else c[a] for a in range(self.config.nb_actions)]  # mask
            action = random.choice(np.where(c == np.max(c))[0])
        else:
            action = random.choice([a for a in range(self.config.nb_actions) if a not in self.invalid_actions])

        return action

    def _simulation(self, env: EnvBase, state: str, player_index, depth: int = 0):
        if depth >= env.max_episode_steps:  # for safety
            return 0

        next_player_indices = env.get_next_player_indices()
        assert player_index in next_player_indices
        action = self._select_action(env, state, player_index)

        # --- steps
        reward = 0
        n_state = state
        while True:
            actions = [self._select_action(env, n_state, idx) for idx in next_player_indices]
            n_state, rewards, done, next_player_indices, _ = env.step(actions)
            n_state = to_str_observation(n_state)
            reward += rewards[player_index]

            if player_index in next_player_indices:
                break
            if done:
                break

        if done:
            pass  # 終了
        elif self.parameter.N[state][action] < self.config.action_select_threshold:
            # 閾値以下はロールアウト
            reward += self._rollout(env, player_index, next_player_indices)
        else:
            # 展開
            reward += self._simulation(env, n_state, player_index, depth + 1)

        # 結果を記録
        self.parameter.N[state][action] += 1
        self.parameter.W[state][action] += reward

        if self.distributed:
            self.remote_memory.add(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                }
            )

        return reward * self.config.gamma  # 割り引いて前に伝搬

    def _select_action(self, env, state, idx):
        self.parameter.init_state(state)
        invalid_actions = env.get_invalid_actions(idx)

        # --- UCBに従ってアクションを選択
        N = np.sum(self.parameter.N[state])
        ucb_list = []
        for a in range(self.config.nb_actions):
            if a in invalid_actions:
                ucb = -np.inf
            else:
                n = self.parameter.N[state][a]
                if n == 0:  # 1度は選んでほしい
                    ucb = np.inf
                else:
                    # UCB値を計算
                    cost = self.config.uct_c * np.sqrt(np.log(N) / n)
                    ucb = self.parameter.W[state][a] / n + cost
            ucb_list.append(ucb)
        action = random.choice(np.where(ucb_list == np.max(ucb_list))[0])
        return action

    # ロールアウト
    def _rollout(self, env: EnvBase, player_index, next_player_indices):
        step = 0
        done = False
        reward = 0
        while not done and step < env.max_episode_steps:
            step += 1

            # step, random
            state, rewards, done, next_player_indices, _ = env.step(env.sample(next_player_indices))
            reward = rewards[player_index] + self.config.gamma * reward

        return reward

    def on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        player_index: int,
        env: EnvBase,
    ):
        self.state = to_str_observation(next_state)
        self.invalid_actions = env.get_invalid_actions(player_index)
        return {}

    def render(self, env: EnvBase) -> None:
        self.parameter.init_state(self.state)
        maxa = np.argmax(self.parameter.N[self.state])
        for a in range(self.config.nb_actions):
            if a == maxa:
                s = "*"
            else:
                s = " "
            q = self.parameter.W[self.state][a]
            c = self.parameter.N[self.state][a]
            if c != 0:
                q /= c
            print(f"{s}{env.action_to_str(a)}: {q:9.5f}({c:7d})")
