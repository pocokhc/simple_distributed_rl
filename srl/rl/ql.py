import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np
from srl.base.rl import RLParameter, RLTrainer, RLWorker, TableConfig
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

    # common
    batch_size: int = 1
    memory_warmup_size: int = 10

    def __post_init__(self):
        super().__init__(self.batch_size, self.memory_warmup_size)

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

    def restore(self, data: Optional[Any]) -> None:
        if data is None:
            return
        self.Q = json.loads(data)

    def backup(self):
        return json.dumps(self.Q)

    # ----------------------------------------

    def _get_q(self, state, valid_actions):
        if state not in self.Q:
            self.Q[state] = [0 if a in valid_actions else -np.inf for a in range(self.config.nb_actions)]
        return self.Q[state]

    def _calc_target_q(self, n_state, n_valid_actions, reward, done):
        n_q = self._get_q(n_state, n_valid_actions)

        if done:
            target_q = reward
        else:
            target_q = reward + self.config.gamma * max(n_q)
        return target_q


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)

    def train_on_batchs(self, batchs: list, weights: list[float]):
        priorities = []
        td_error_mean = 0
        for i in range(self.config.batch_size):

            # データ形式を変形
            s = batchs[i]["state"]
            n_s = batchs[i]["next_state"]
            action = batchs[i]["action"]
            reward = batchs[i]["reward"]
            done = batchs[i]["done"]
            valid_actions = batchs[i]["valid_actions"]
            next_valid_actions = batchs[i]["next_valid_actions"]
            weight = weights[i]

            q = self.parameter._get_q(s, valid_actions)

            # Q値の計算
            target_q = self.parameter._calc_target_q(n_s, next_valid_actions, reward, done)
            td_error = target_q - q[action]
            q[action] += self.config.lr * td_error * weight

            priority = abs(td_error) + 0.0001
            priorities.append(priority)

            td_error_mean += td_error

        return priorities, {
            "td_error": td_error_mean / self.config.batch_size,
            "Q": len(self.parameter.Q),
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)

        self.step = 0

    def on_reset(self, state: np.ndarray, valid_actions: list[int]) -> None:
        pass

    def policy(self, state: np.ndarray, valid_actions: list[int]) -> tuple[int, Any]:
        s = str(state.tolist())

        if self.training:
            epsilon = self.config.epsilon
        else:
            epsilon = self.config.test_epsilon

        q = self.parameter._get_q(s, valid_actions)

        if random.random() < epsilon:
            # epsilonより低いならランダムに移動
            action = random.choice(valid_actions)
        else:
            # valid actionsでfilter
            q = np.asarray([val if a in valid_actions else -np.inf for a, val in enumerate(q)])

            # 最大値を選ぶ（複数あればランダム）
            action = random.choice(np.where(q == q.max())[0])

        return action, (action, q[action])

    def on_step(
        self,
        state: np.ndarray,
        action_: Any,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        valid_actions: list[int],
        next_valid_actions: list[int],
    ):
        if not self.training:
            return {}

        action = action_[0]
        q = action_[1]
        s = str(state.tolist())
        n_s = str(next_state.tolist())

        # priority を計算
        target_q = self.parameter._calc_target_q(n_s, next_valid_actions, reward, done)
        priority = abs(target_q - q) + 0.0001

        batch = {
            "state": s,
            "next_state": n_s,
            "action": action,
            "reward": reward,
            "done": done,
            "valid_actions": valid_actions,
            "next_valid_actions": next_valid_actions,
        }
        return (batch, priority, {"Q": len(self.parameter.Q)})

    def render(self, state: np.ndarray, valid_actions: list[int]) -> None:
        s = str(state.tolist())
        q = self.parameter._get_q(s, valid_actions)
        maxa = np.argmax(q)
        for a in range(self.config.nb_actions):
            if a == maxa:
                s = "*"
            else:
                s = " "
            s += f"{a:3d}: {q[a]:7.5f}"
            print(s)
