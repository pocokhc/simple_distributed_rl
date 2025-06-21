import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from srl.base.env.env_run import EnvRun
from srl.base.rl.algorithms.base_ql import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.rl.memories.single_use_buffer import RLSingleUseBuffer


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

    def get_name(self) -> str:
        return "MCTS"

    def use_backup_restore(self) -> bool:
        return True

    def use_update_parameter_from_worker(self) -> bool:
        return True


register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


class Memory(RLSingleUseBuffer):
    pass


class Parameter(RLParameter[Config]):
    def setup(self):
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

    def update_from_worker_parameter(self, worker_parameger: "Parameter"):
        self.call_restore(worker_parameger.call_backup())

    # ------------------------

    def init_state(self, state):
        if state not in self.N:
            self.W[state] = [0 for _ in range(self.config.action_space.n)]
            self.N[state] = [0 for _ in range(self.config.action_space.n)]


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def train(self) -> None:
        self.train_count += 1


class Worker(RLWorker[Config, Parameter, Memory]):
    def policy(self, worker) -> int:
        self.state = self.config.observation_space.to_str(worker.state)
        self.parameter.init_state(self.state)

        if self.training:
            dat = worker.env.backup()
            for _ in range(self.config.num_simulations):
                self._simulation(worker.env, self.state)
                worker.env.restore(dat)

        # 試行回数のもっとも多いアクションを採用
        c = self.parameter.N[self.state]
        c = [-np.inf if a in worker.invalid_actions else c[a] for a in range(self.config.action_space.n)]  # mask
        action = int(np.random.choice(np.where(c == np.max(c))[0]))

        self.info["size"] = len(self.parameter.N)
        return action

    def _simulation(self, env: EnvRun, state: str, depth: int = 0):
        if depth >= env.max_episode_steps:  # for safety
            return 0

        # actionを選択
        uct_list = self._calc_uct(state, env.get_invalid_actions())
        action = int(np.random.choice(np.where(uct_list == np.max(uct_list))[0]))

        if self.parameter.N[state][action] < self.config.expansion_threshold:
            # アクション回数がすくないのでロールアウト
            reward = self._rollout(env)
        else:
            # 1step実行
            player_index = env.next_player
            n_state: Any = env.step_from_rl(action, self.config)
            reward = env.rewards[player_index]

            if env.done:
                pass  # 終了
            else:
                n_state = self.config.observation_space.to_str(n_state)

                enemy_turn = player_index != env.next_player

                # expansion
                n_reward = self._simulation(env, n_state)

                # 次が相手のターンの報酬は最小になってほしいので-をかける
                if enemy_turn:
                    n_reward = -n_reward

                # 割引報酬
                reward = reward + self.config.discount * n_reward

        self.parameter.N[state][action] += 1
        self.parameter.W[state][action] += reward

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
        uct_list = self._calc_uct(self.state, worker.invalid_actions)

        def _render_sub(a: int) -> str:
            q = self.parameter.W[self.state][a]
            c = self.parameter.N[self.state][a]
            if c != 0:
                q /= c
            return f"{c:7d}(N), {q:9.4f}(Q), {uct_list[a]:.5f}(UCT)"

        worker.print_discrete_action_info(int(maxa), _render_sub)
