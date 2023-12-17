import json
import random
from dataclasses import dataclass

import numpy as np

from srl.base.define import RLBaseTypes
from srl.base.rl.algorithms.discrete_action import DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.config import RLConfig
from srl.base.rl.registration import register
from srl.rl.memories.sequence_memory import SequenceMemory


@dataclass
class Config(RLConfig):
    epsilon: float = 0.1
    test_epsilon: float = 0
    gamma: float = 0.9
    lr: float = 0.1

    @property
    def base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    @property
    def base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    def get_use_framework(self) -> str:
        return ""

    def getName(self) -> str:
        return "MyRL"


class Memory(SequenceMemory):
    pass


class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.Q = {}  # Q学習用のテーブル

    def call_restore(self, data, **kwargs) -> None:
        self.Q = json.loads(data)

    def call_backup(self, **kwargs):
        return json.dumps(self.Q)

    # Q値を取得する関数
    def get_action_values(self, state: str):
        if state not in self.Q:
            self.Q[state] = [0] * self.config.action_num
        return self.Q[state]


class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        batchs = self.memory.sample(self.batch_size, self.train_count)

        td_error = 0
        for batch in batchs:
            # 各batch毎にQテーブルを更新する
            s = batch["state"]
            n_s = batch["next_state"]
            action = batch["action"]
            reward = batch["reward"]
            done = batch["done"]

            q = self.parameter.get_action_values(s)
            n_q = self.parameter.get_action_values(n_s)

            if done:
                target_q = reward
            else:
                target_q = reward + self.config.gamma * max(n_q)

            td_error = target_q - q[action]
            q[action] += self.config.lr * td_error

            td_error += td_error
            self.train_count += 1  # 学習回数

        if len(batchs) > 0:
            td_error /= len(batchs)

        # 学習結果(任意)
        self.train_info = {
            "Q": len(self.parameter.Q),
            "td_error": td_error,
        }


class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

    def call_on_reset(self, state: np.ndarray, invalid_actions: list[int]) -> dict:
        return {}

    def call_policy(self, state: np.ndarray, invalid_actions: list[int]) -> tuple[int, dict]:
        self.state = str(state.tolist())

        # 学習中かどうかで探索率を変える
        if self.training:
            epsilon = self.config.epsilon
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            # epsilonより低いならランダムに移動
            self.action = self.sample_action()
        else:
            q = self.parameter.get_action_values(self.state)
            q = np.asarray(q)

            # 最大値を選ぶ（複数あればランダム）
            self.action = np.random.choice(np.where(q == q.max())[0])

        return int(self.action), {}

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: list[int],
    ) -> dict:
        if not self.training:
            return {}

        batch = {
            "state": self.state,
            "next_state": str(next_state.tolist()),
            "action": self.action,
            "reward": reward,
            "done": done,
        }
        self.memory.add(batch)  # memoryはaddのみ
        return {}

    # 強化学習の可視化用、今回ですとQテーブルを表示しています。
    def render_terminal(self, worker, **kwargs) -> None:
        q = self.parameter.get_action_values(self.state)
        maxa = np.argmax(q)
        for a in range(self.config.action_num):
            if a == maxa:
                s = "*"
            else:
                s = " "
            s += f"{worker.env.action_to_str(a)}: {q[a]:7.5f}"
            print(s)


# ---------------------------------
# 登録
# ---------------------------------
register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


# ---------------------------------
# テスト
# ---------------------------------
from srl.test import TestRL

tester = TestRL()
tester.simple_check(Config())


# ---------------------------------
# Grid環境の学習
# ---------------------------------
import srl

runner = srl.Runner(srl.EnvConfig("Grid"), Config(lr=0.001))

# --- train
runner.train(timeout=10)

# --- test
rewards = runner.evaluate(max_episodes=100)
print("100エピソードの平均結果", np.mean(rewards))

runner.render_terminal()

runner.animation_save_gif("_MyRL-Grid.gif", render_scale=2)
