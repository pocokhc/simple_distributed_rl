import json
import random
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from srl.base.define import RLObservationType
from srl.base.env.base import EnvRun
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig
from srl.base.rl.base import RLParameter, RLTrainer, RLWorker
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import SequenceRemoteMemory
from srl.rl.functions.common import render_discrete_action, to_str_observation


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    simulation_times: int = 10
    action_select_threshold: int = 5
    gamma: float = 1.0  # 割引率
    uct_c: float = np.sqrt(2.0)

    def __post_init__(self):
        super().__init__()

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.DISCRETE

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
            self.W[state] = [0 for _ in range(self.config.action_num)]
            self.N[state] = [0 for _ in range(self.config.action_num)]


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

    def _on_reset(self, state: np.ndarray, player_index: int, env: EnvRun) -> None:
        self.state = to_str_observation(state)
        self.invalid_actions = self.get_invalid_actions(env, player_index)

    def _policy(self, _state: np.ndarray, player_index: int, env: EnvRun) -> int:
        state = to_str_observation(_state)

        if self.training:
            dat = env.backup()
            for _ in range(self.config.simulation_times):
                self._simulation(env, state, player_index)
                env.restore(dat)

        # 試行回数のもっとも多いアクションを採用
        if state in self.parameter.N:
            c = self.parameter.N[state]
            c = [-np.inf if a in self.invalid_actions else c[a] for a in range(self.config.action_num)]  # mask
            action = random.choice(np.where(c == np.max(c))[0])
        else:
            action = random.choice([a for a in range(self.config.action_num) if a not in self.invalid_actions])

        return action

    def _simulation(self, env: EnvRun, state: str, player_index, depth: int = 0):
        if depth >= env.max_episode_steps:  # for safety
            return 0

        next_player_indices = env.get_next_player_indices()
        assert player_index in next_player_indices
        action = self._select_action(env, state, player_index)

        # --- steps  TODO simulation stepが複雑すぎるので何か簡単にできる方法
        reward = 0
        n_state = state
        while True:
            actions = [self._select_action(env, n_state, idx) for idx in next_player_indices]
            n_state, rewards, done, next_player_indices, _ = env.step(actions)
            n_state = self.config.env_observation_space.observation_discrete_encode(n_state)
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
        invalid_actions = env.get_invalid_actions(idx)
        ucb_list = self._calc_ucb(state, invalid_actions)
        action = random.choice(np.where(ucb_list == np.max(ucb_list))[0])
        return action

    def _calc_ucb(self, state, invalid_actions):
        self.parameter.init_state(state)

        # --- UCBに従ってアクションを選択
        N = np.sum(self.parameter.N[state])
        ucb_list = []
        for a in range(self.config.action_num):
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
        return ucb_list

    # ロールアウト
    def _rollout(self, env: EnvRun, player_index, next_player_indices):
        step = 0
        done = False
        reward = 0
        while not done and step < env.max_episode_steps:
            step += 1

            # step, random
            state, rewards, done, next_player_indices, _ = env.step(env.sample(next_player_indices))
            reward = rewards[player_index] + self.config.gamma * reward

        return reward

    def _on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        player_index: int,
        env: EnvRun,
    ):
        self.state = to_str_observation(next_state)
        self.invalid_actions = self.get_invalid_actions(env, player_index)
        return {}

    def call_render(self, env: EnvRun) -> None:
        self.parameter.init_state(self.state)
        maxa = np.argmax(self.parameter.N[self.state])
        ucb_list = self._calc_ucb(self.state, self.invalid_actions)

        def _render_sub(a: int) -> str:
            q = self.parameter.W[self.state][a]
            c = self.parameter.N[self.state][a]
            if c != 0:
                q /= c
            return f"{q:9.5f}({c:7d}), ucb {ucb_list[a]:.5f}"

        render_discrete_action(self.invalid_actions, maxa, env, _render_sub)
