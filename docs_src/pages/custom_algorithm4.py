import json
import pickle
import random
from dataclasses import dataclass

import numpy as np

from srl.base.define import RLBaseActTypes, RLBaseObsTypes
from srl.base.rl.config import RLConfig
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker import RLWorker
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.discrete import DiscreteSpace


@dataclass
class Config(RLConfig[DiscreteSpace, ArrayDiscreteSpace]):
    epsilon: float = 0.1
    test_epsilon: float = 0
    gamma: float = 0.9
    lr: float = 0.1

    def get_base_action_type(self) -> RLBaseActTypes:
        return RLBaseActTypes.DISCRETE

    def get_base_observation_type(self) -> RLBaseObsTypes:
        return RLBaseObsTypes.DISCRETE

    def get_name(self) -> str:
        return "MyRL"


# 継承する場合
# from srl.rl.memories.single_use_buffer import RLSingleUseBuffer
# class Memory(RLSingleUseBuffer):
#     pass


class Memory(RLMemory[Config]):
    def setup(self):
        self.buffer = []
        self.register_worker_func(self.add, pickle.dumps)
        self.register_trainer_recv_func(self.sample)

    def length(self) -> int:
        return len(self.buffer)

    def add(self, batch, serialized: bool = False) -> None:
        if serialized:
            batch = pickle.loads(batch)
        self.buffer.append(batch)

    def sample(self):
        if len(self.buffer) == 0:
            return None
        buffer = self.buffer
        self.buffer = []
        return buffer

    def call_backup(self, **kwargs):
        return self.buffer[:]

    def call_restore(self, data, **kwargs) -> None:
        self.buffer = data[:]


class Parameter(RLParameter[Config]):
    def setup(self):
        self.Q = {}  # Q学習用のテーブル

    def call_restore(self, data, **kwargs) -> None:
        self.Q = json.loads(data)

    def call_backup(self, **kwargs):
        return json.dumps(self.Q)

    # Q値を取得する関数
    def get_action_values(self, state: str):
        if state not in self.Q:
            self.Q[state] = [0] * self.config.action_space.n
        return self.Q[state]


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        pass

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return

        td_error = 0
        for batch in batches:
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

        if len(batches) > 0:
            td_error /= len(batches)

        # 学習結果(任意)
        self.info["Q"] = len(self.parameter.Q)
        self.info["td_error"] = td_error


class Worker(RLWorker[Config, Parameter, Memory]):
    def on_setup(self, worker, context) -> None:
        pass

    def on_teardown(self, worker) -> None:
        pass

    def on_reset(self, worker):
        pass

    def policy(self, worker) -> int:
        self.state = self.config.observation_space.to_str(worker.state)

        # 学習中かどうかで探索率を変える
        if self.training:
            epsilon = self.config.epsilon
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            # epsilonより低いならランダムに移動
            action = self.sample_action()
        else:
            q = self.parameter.get_action_values(self.state)
            q = np.asarray(q)

            # 最大値を選ぶ（複数あればランダム）
            action = np.random.choice(np.where(q == q.max())[0])

        return int(action)

    def on_step(self, worker):
        if not self.training:
            return

        self.memory.add(
            {
                "state": self.state,
                "next_state": self.config.observation_space.to_str(worker.state),
                "action": worker.action,
                "reward": worker.reward,
                "done": worker.terminated,
            }
        )

    # 強化学習の可視化用、今回ですとQテーブルを表示しています。
    def render_terminal(self, worker, **kwargs):
        q = self.parameter.get_action_values(self.state)
        maxa = np.argmax(q)
        for a in range(self.config.action_space.n):
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


def test():
    from srl.test.rl import test_rl

    test_rl(Config())


def main():
    # --- Grid環境の学習
    import srl

    runner = srl.Runner(srl.EnvConfig("Grid"), Config(lr=0.001))

    # --- train
    runner.train(timeout=10)
    runner.train_mp(timeout=10)

    # --- test
    rewards = runner.evaluate(max_episodes=100)
    print("100エピソードの平均結果", np.mean(rewards))

    runner.render_terminal()

    runner.animation_save_gif("_MyRL-Grid.gif", render_scale=2)


if __name__ == "__main__":
    # test()
    main()
