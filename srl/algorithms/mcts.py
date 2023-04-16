import json
from dataclasses import dataclass
from typing import Any, Tuple, cast

import numpy as np

from srl.base.define import RLObservationType
from srl.base.env.base import EnvRun
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig
from srl.base.rl.algorithms.modelbase import ModelBaseWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import SequenceRemoteMemory
from srl.rl.functions.common import render_discrete_action, to_str_observation


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):
    num_simulations: int = 10
    expansion_threshold: int = 5
    discount: float = 1.0
    uct_c: float = np.sqrt(2.0)

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.DISCRETE

    def getName(self) -> str:
        return "MCTS"


register(
    Config(),
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

    def call_restore(self, data: Any, **kwargs) -> None:
        d = json.loads(data)
        self.N = d[0]
        self.W = d[1]

    def call_backup(self, **kwargs):
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
            self.parameter.init_state(state)

            self.parameter.N[state][action] += 1
            self.parameter.W[state][action] += reward
            self.train_count += 1
        return {"size": len(self.parameter.N)}


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(ModelBaseWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

    def call_on_reset(self, state: np.ndarray, env: EnvRun, worker) -> dict:
        return {}

    def call_policy(self, _state: np.ndarray, env: EnvRun, worker) -> Tuple[int, dict]:
        self.state = to_str_observation(_state)
        self.invalid_actions = self.get_invalid_actions()
        self.parameter.init_state(self.state)

        if self.training:
            dat = env.backup()
            for _ in range(self.config.num_simulations):
                self._simulation(env, self.state, self.invalid_actions)
                env.restore(dat)

        # 試行回数のもっとも多いアクションを採用
        c = self.parameter.N[self.state]
        c = [-np.inf if a in self.invalid_actions else c[a] for a in range(self.config.action_num)]  # mask
        action = int(np.random.choice(np.where(c == np.max(c))[0]))

        return action, {}

    def _simulation(self, env: EnvRun, state: str, invalid_actions, depth: int = 0):
        if depth >= env.max_episode_steps:  # for safety
            return 0

        player_index = env.next_player_index

        # actionを選択
        uct_list = self._calc_uct(state, invalid_actions)
        action = np.random.choice(np.where(uct_list == np.max(uct_list))[0])

        if self.parameter.N[state][action] < self.config.expansion_threshold:
            # アクション回数がすくないのでロールアウト
            reward = self._rollout(env, player_index)
        else:
            # 1step実行
            n_state, rewards, done = self.env_step(env, action)
            reward = rewards[player_index]

            if done:
                pass  # 終了
            else:
                n_state = to_str_observation(n_state)
                n_invalid_actions = self.get_invalid_actions(env)

                enemy_turn = player_index != env.next_player_index

                # expansion
                n_reward = self._simulation(env, n_state, n_invalid_actions)

                # 次が相手のターンの報酬は最小になってほしいので-をかける
                if enemy_turn:
                    n_reward = -n_reward

                # 割引報酬
                reward = reward + self.config.discount * n_reward

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

        return reward

    def _calc_uct(self, state, invalid_actions):
        self.parameter.init_state(state)

        N = np.sum(self.parameter.N[state])
        uct_list = []
        for a in range(self.config.action_num):
            if a in invalid_actions:
                uct = -np.inf
            else:
                n = self.parameter.N[state][a]
                if n == 0:  # 1度は選んでほしい
                    uct = np.inf
                else:
                    # UCT値を計算
                    q = self.parameter.W[state][a] / n
                    cost = self.config.uct_c * np.sqrt(np.log(N) / n)
                    uct = q + cost
            uct_list.append(uct)
        return uct_list

    # ロールアウト
    def _rollout(self, env: EnvRun, player_index):
        rewards = []
        while not env.done:
            env.step(env.sample())
            rewards.append(env.step_rewards[player_index])

        # 割引報酬
        reward = 0
        for r in reversed(rewards):
            reward = r + self.config.discount * reward
        return reward

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        env: EnvRun,
        worker,
    ) -> dict:
        return {}

    def render_terminal(self, env, worker, **kwargs) -> None:
        self.parameter.init_state(self.state)
        maxa = np.argmax(self.parameter.N[self.state])
        uct_list = self._calc_uct(self.state, self.invalid_actions)

        def _render_sub(a: int) -> str:
            q = self.parameter.W[self.state][a]
            c = self.parameter.N[self.state][a]
            if c != 0:
                q /= c
            return f"{c:7d}(N), {q:9.4f}(Q), {uct_list[a]:.5f}(UCT)"

        render_discrete_action(self.invalid_actions, maxa, env, _render_sub)
