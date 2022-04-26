import json
import random
from dataclasses import dataclass
from typing import Any, List, cast

import numpy as np
from srl.base.rl import RLParameter, RLRemoteMemory, RLTrainer, RLWorker, TableConfig
from srl.base.rl.env_for_rl import EnvForRL
from srl.base.rl.registory import register


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(TableConfig):

    simulation_times: int = 1
    action_select_threshold: int = 10
    gamma: float = 1.0  # 割引率
    uct_c: float = np.sqrt(2.0)

    @staticmethod
    def getName() -> str:
        return "MCTS"


register(Config, __name__)


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
# RemoteMemory
# ------------------------------------------------------
class RemoteMemory(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.buffer = []

    def length(self) -> int:
        return len(self.buffer)

    def restore(self, data: Any) -> None:
        self.buffer = data

    def backup(self):
        return self.buffer

    # ------------------------

    def add(self, batch: Any) -> None:
        self.buffer.append(batch)

    def get(self):
        buffer = self.buffer
        self.buffer = []
        return buffer


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.memory = cast(RemoteMemory, self.memory)

        self.train_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):

        batchs = self.memory.get()
        for batch in batchs:
            state = batch["state"]
            action = batch["action"]
            reward = batch["reward"]

            self.parameter.init_state(state)
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
        self.memory = cast(RemoteMemory, self.memory)

    def on_reset(self, state: np.ndarray, invalid_actions: List[int], env: EnvForRL) -> None:
        pass

    def policy(self, state: np.ndarray, invalid_actions: List[int], env: EnvForRL):
        s = str(state.tolist())

        if self.training:
            # シミュレーション
            for _ in range(self.config.simulation_times):
                save_state = env.backup()
                self._simulation(env, str(state.tolist()), invalid_actions)
                env.restore(save_state)

        # 試行回数のもっとも多いアクションを採用
        if s in self.parameter.N:
            c = self.parameter.N[s]
            c = [c[a] if a in invalid_actions else -np.inf for a in range(self.config.nb_actions)]  # mask
            action = random.choice(np.where(c == np.max(c))[0])
        else:
            action = random.choice(invalid_actions)

        return action, action

    def on_step(
        self,
        state: np.ndarray,
        action: Any,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        invalid_actions: List[int],
        next_invalid_actions: List[int],
        env: EnvForRL,
    ):
        return {}

    def _simulation(self, env: EnvForRL, state: str, invalid_actions, depth: int = 0):
        if depth >= env.max_episode_steps:  # for safety
            return 0
        self.parameter.init_state(state)

        # --- UCBに従ってアクションを選択
        N = np.sum(self.parameter.N[state])
        ucb_list = []
        for a in range(self.config.nb_actions):
            if a not in invalid_actions:
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

        # --- step
        if self.parameter.N[state][action] < self.config.action_select_threshold:
            # 閾値以下はロールアウト
            reward = self._rollout(env)
        else:
            # step
            n_state, reward, done, _ = env.step(action)
            n_state = str(n_state.tolist())
            n_invalid_actions = env.fetch_invalid_actions()

            if done:
                pass  # 終了(終了時の報酬が結果)
            else:
                # 展開
                reward += self._simulation(env, n_state, n_invalid_actions, depth + 1)

        # 結果を記録
        batch = {
            "state": state,
            "action": action,
            "reward": reward,
        }
        self.memory.add(batch)

        return reward * self.config.gamma  # 割り引いて前に伝搬

    # ロールアウト
    def _rollout(self, env: EnvForRL):
        step = 0
        done = False
        reward = 0
        while not done and step < env.max_episode_steps:
            step += 1

            # ランダム
            invalid_actions = env.fetch_invalid_actions()
            action = random.choice(invalid_actions)

            # step
            state, _reward, done, _ = env.step(action)
            reward = _reward + self.config.gamma * reward

        return reward

    def render(
        self,
        state: np.ndarray,
        invalid_actions: List[int],
        env: EnvForRL,
    ) -> None:
        s = str(state.tolist())
        for a in range(self.config.nb_actions):
            if s in self.parameter.W:
                q = self.parameter.W[s][a]
                c = self.parameter.N[s][a]
                if c != 0:
                    q /= c
            else:
                q = 0
                c = 0
            print(f"{action_to_str(a)}: {q:7.3f}({c:7d})|")


if __name__ == "__main__":
    pass
