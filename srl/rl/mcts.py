import json
import random
from dataclasses import dataclass
from typing import Any, List, cast

import numpy as np
import srl
from srl.base.env.env_for_rl import EnvForRL
from srl.base.rl.algorithms.table import TableConfig
from srl.base.rl.base import RLParameter, RLTrainer, RLWorker
from srl.base.rl.registration import register
from srl.rl.functions.common import to_str_observaten
from srl.rl.remote_memory.sequence_memory import SequenceRemoteMemory


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(TableConfig):

    simulation_times: int = 10
    action_select_threshold: int = 10
    gamma: float = 1.0  # 割引率
    uct_c: float = np.sqrt(2.0)

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
        self.env = srl.envs.make(self.config.env_config, self.config)

    def get_train_count(self):
        return self.train_count

    def train(self):
        # シミュレーションがアクション前ではなくアクション後になっている点が異なっています

        batchs = self.remote_memory.sample()
        for batch in batchs:
            state = batch["state"]
            invalid_actions = batch["invalid_actions"]
            self.env.restore(batch["env"])

            # シミュレーション
            for _ in range(self.config.simulation_times):
                save_state = self.env.backup()
                self._simulation(self.env, state, invalid_actions)
                self.env.restore(save_state)

                self.train_count += 1

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
            n_state, rewards, done, next_player_indices, done, _ = env.step(action)
            assert len(next_player_indices) == 1  # とりあえず一人のみ対応 TODO

            n_state = to_str_observaten(n_state)
            n_invalid_actions = env.fetch_invalid_actions()

            if done:
                pass  # 終了(終了時の報酬が結果)
            else:
                # 展開
                reward += self._simulation(env, n_state, n_invalid_actions, depth + 1)

        # 結果を記録
        self.parameter.N[state][action] += 1
        self.parameter.W[state][action] += reward

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


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

    def on_reset(self, state: np.ndarray, invalid_actions: List[int], env: EnvForRL) -> None:
        self.remote_memory.add(
            {
                "state": to_str_observaten(state),
                "action": invalid_actions,
                "env": env.backup(),
            }
        )

    def policy(self, state: np.ndarray, invalid_actions: List[int], env: EnvForRL) -> int:
        s = to_str_observaten(state)

        # 試行回数のもっとも多いアクションを採用
        if s in self.parameter.N:
            c = self.parameter.N[s]
            c = [c[a] if a in invalid_actions else -np.inf for a in range(self.config.nb_actions)]  # mask
            action = random.choice(np.where(c == np.max(c))[0])
        else:
            action = random.choice([a for a in range(self.config.nb_actions) if a not in invalid_actions])

        return action

    def on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
        env: EnvForRL,
    ):
        self.remote_memory.add(
            {
                "state": to_str_observaten(next_state),
                "action": next_invalid_actions,
                "env": env.backup(),
            }
        )
        return {}

    def render(self, env: EnvForRL) -> None:
        s = env.to_str_observaten(state)
        for a in range(self.config.nb_actions):
            if s in self.parameter.W:
                q = self.parameter.W[s][a]
                c = self.parameter.N[s][a]
                if c != 0:
                    q /= c
            else:
                q = 0
                c = 0
            print(f"{env.action_to_str(a)}: {q:7.3f}({c:7d})|")


if __name__ == "__main__":
    pass
