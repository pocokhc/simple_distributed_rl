import json
import logging
import random
from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import numpy as np
from srl.base.rl import RLParameter, RLRemoteMemory, RLTrainer, RLWorker, TableConfig
from srl.rl.registory import register

logger = logging.getLogger(__name__)


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(TableConfig):

    epsilon: float = 0.1
    test_epsilon: float = 0
    gamma: float = 0.9  # 割引率
    lr: float = 0.1  # 学習率

    @staticmethod
    def getName() -> str:
        return "QL"


register(Config, __name__)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.Q = {}

    def restore(self, data: Any) -> None:
        self.Q = json.loads(data)

    def backup(self):
        return json.dumps(self.Q)

    # ----------------------------------------

    def _get_q(self, state, valid_actions):
        if state not in self.Q:
            self.Q[state] = [0 if a in valid_actions else -np.inf for a in range(self.config.nb_actions)]
        return self.Q[state]


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

    # --------------------

    def add(self, batch: Any) -> None:
        self.buffer.append(batch)

    def sample(self):
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

        batchs = self.memory.sample()
        td_error = 0
        for batch in batchs:

            # データ形式を変形
            s = batch["state"]
            n_s = batch["next_state"]
            action = batch["action"]
            reward = batch["reward"]
            done = batch["done"]
            valid_actions = batch["valid_actions"]
            next_valid_actions = batch["next_valid_actions"]

            q = self.parameter._get_q(s, valid_actions)
            n_q = self.parameter._get_q(n_s, next_valid_actions)

            if done:
                target_q = reward
            else:
                target_q = reward + self.config.gamma * max(n_q)

            td_error = target_q - q[action]
            q[action] += self.config.lr * td_error

            td_error += td_error
            self.train_count += 1

        if len(batchs) > 0:
            td_error /= len(batchs)

        return {
            "Q": len(self.parameter.Q),
            "td_error": td_error,
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.memory = cast(RemoteMemory, self.memory)

    def on_reset(self, state: np.ndarray, valid_actions: List[int]) -> None:
        if self.training:
            self.epsilon = self.config.epsilon
        else:
            self.epsilon = self.config.test_epsilon

    def policy(self, state: np.ndarray, valid_actions: List[int]) -> Tuple[int, Any]:
        s = str(state.tolist())

        if random.random() < self.epsilon:
            # epsilonより低いならランダムに移動
            action = random.choice(valid_actions)
        else:
            q = self.parameter._get_q(s, valid_actions)

            # valid actionsでfilter
            q = np.asarray([val if a in valid_actions else -np.inf for a, val in enumerate(q)])

            # 最大値を選ぶ（複数あればランダム）
            action = random.choice(np.where(q == q.max())[0])

        return action, action

    def on_step(
        self,
        state: np.ndarray,
        action: Any,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        valid_actions: List[int],
        next_valid_actions: List[int],
    ):
        if not self.training:
            return {}

        batch = {
            "state": str(state.tolist()),
            "next_state": str(next_state.tolist()),
            "action": action,
            "reward": reward,
            "done": done,
            "valid_actions": valid_actions,
            "next_valid_actions": next_valid_actions,
        }
        self.memory.add(batch)
        return {}

    def render(self, state: np.ndarray, valid_actions: List[int], action_to_str) -> None:
        s = str(state.tolist())
        q = self.parameter._get_q(s, valid_actions)
        maxa = np.argmax(q)
        for a in range(self.config.nb_actions):
            if a == maxa:
                s = "*"
            else:
                s = " "
            s += f"{action_to_str(a)}: {q[a]:7.5f}"
            print(s)
