import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from srl.base.env.env_run import EnvRun
from srl.base.rl.algorithms.base_ql import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.rl import functions as funcs
from srl.rl.memories.sequence_memory import SequenceMemory


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig):
    #: シミュレーション回数
    num_simulations: int = 10
    #: 展開の閾値
    expansion_threshold: int = 5
    #: 割引率
    discount: float = 1.0
    #: UCT C
    uct_c: float = np.sqrt(2.0)

    def get_framework(self) -> str:
        return ""

    def get_name(self) -> str:
        return "MCTS"

    def get_used_backup_restore(self) -> bool:
        return True


register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


# ------------------------------------------------------
# Memory
# ------------------------------------------------------
class Memory(SequenceMemory):
    pass


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter[Config]):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

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
            self.W[state] = [0 for _ in range(self.config.action_space.n)]
            self.N[state] = [0 for _ in range(self.config.action_space.n)]


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        batchs = self.memory.sample()

        for batch in batchs:
            state = batch["state"]
            action = batch["action"]
            reward = batch["reward"]
            self.parameter.init_state(state)

            self.parameter.N[state][action] += 1
            self.parameter.W[state][action] += reward

            self.train_count += 1

        self.info["size"] = len(self.parameter.N)


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

    def policy(self, worker) -> int:
        self.state = self.config.observation_space.to_str(worker.state)
        self.invalid_actions = worker.get_invalid_actions()
        self.parameter.init_state(self.state)

        if self.training:
            dat = worker.env.backup()
            for _ in range(self.config.num_simulations):
                self._simulation(worker.env, self.state)
                worker.env.restore(dat)

        # 試行回数のもっとも多いアクションを採用
        c = self.parameter.N[self.state]
        c = [-np.inf if a in self.invalid_actions else c[a] for a in range(self.config.action_space.n)]  # mask
        action = int(np.random.choice(np.where(c == np.max(c))[0]))

        return action

    def _simulation(self, env: EnvRun, state: str, depth: int = 0):
        if depth >= env.max_episode_steps:  # for safety
            return 0

        # actionを選択
        uct_list = self._calc_uct(state, env.get_invalid_actions())
        action = np.random.choice(np.where(uct_list == np.max(uct_list))[0])

        if self.parameter.N[state][action] < self.config.expansion_threshold:
            # アクション回数がすくないのでロールアウト
            reward = self._rollout(env)
        else:
            # 1step実行
            player_index = env.next_player_index
            n_state, rewards = self.worker.env_step(env, action)
            reward = rewards[player_index]

            if env.done:
                pass  # 終了
            else:
                n_state = self.config.observation_space.to_str(n_state)

                enemy_turn = player_index != env.next_player_index

                # expansion
                n_reward = self._simulation(env, n_state)

                # 次が相手のターンの報酬は最小になってほしいので-をかける
                if enemy_turn:
                    n_reward = -n_reward

                # 割引報酬
                reward = reward + self.config.discount * n_reward

        self.parameter.N[state][action] += 1
        self.parameter.W[state][action] += reward

        if self.distributed:
            self.memory.add(
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
        for a in range(self.config.action_space.n):
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
    def _rollout(self, env: EnvRun):
        rewards = []
        while not env.done:
            env.step(env.sample_action())
            rewards.append(env.reward)

        # 割引報酬
        reward = 0
        for r in reversed(rewards):
            reward = r + self.config.discount * reward
        return reward

    def render_terminal(self, worker, **kwargs) -> None:
        self.parameter.init_state(self.state)
        maxa = np.argmax(self.parameter.N[self.state])
        uct_list = self._calc_uct(self.state, self.invalid_actions)

        def _render_sub(a: int) -> str:
            q = self.parameter.W[self.state][a]
            c = self.parameter.N[self.state][a]
            if c != 0:
                q /= c
            return f"{c:7d}(N), {q:9.4f}(Q), {uct_list[a]:.5f}(UCT)"

        funcs.render_discrete_action(int(maxa), self.config.action_space, worker.env, _render_sub)
