import logging
import pickle
import random
from dataclasses import dataclass
from typing import Any, Tuple, Union

import numpy as np

from srl.base.define import InfoType
from srl.base.rl.algorithms.base_ql import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.rl import functions as funcs
from srl.rl.memories.sequence_memory import SequenceMemory
from srl.rl.schedulers.scheduler import SchedulerConfig

logger = logging.getLogger(__name__)


"""
Other
    invalid_actions : o
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig):
    epsilon: Union[float, SchedulerConfig] = 0.1
    test_epsilon: float = 0
    discount: float = 0.9
    lr: Union[float, SchedulerConfig] = 0.1

    def __post_init__(self):
        super().__post_init__()

    def get_framework(self) -> str:
        return ""

    def get_name(self) -> str:
        return "Dyna-Q"

    def get_info_types(self) -> dict:
        return {
            "size": {"type": int, "data": "last"},
            "td_error": {},
        }


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
# 近似モデル
# ------------------------------------------------------
class _A_MDP:
    def __init__(self):
        self.trans = {}  # [state][action][next_state] = 訪れた回数
        self.reward = {}  # [state][action] = 得た報酬の合計
        self.done = {}  # [state][action] = 終了した回数
        self.count = {}  # [state][action] = 訪れた回数

        # サンプリング用に実際にあった履歴を保存
        self.state_action_history = []

    def backup(self):
        return [
            self.trans,
            self.reward,
            self.done,
            self.count,
            self.state_action_history,
        ]

    def restore(self, data):
        d = data
        self.trans = d[0]
        self.reward = d[1]
        self.done = d[2]
        self.count = d[3]
        self.state_action_history = d[4]

    def _init_dict(self, state, action, n_state):
        if state not in self.count:
            self.trans[state] = {}
            self.reward[state] = {}
            self.done[state] = {}
            self.count[state] = {}
        if action not in self.count[state]:
            self.trans[state][action] = {}
            self.reward[state][action] = 0
            self.done[state][action] = 0
            self.count[state][action] = 0
        if n_state is not None and n_state not in self.trans[state][action]:
            self.trans[state][action][n_state] = 0

    def train(self, state, action, n_state, reward, done):
        self._init_dict(state, action, n_state)
        self.state_action_history.append([state, action])

        self.count[state][action] += 1
        self.trans[state][action][n_state] += 1
        self.done[state][action] += 1 if done else 0
        self.reward[state][action] += reward

    # ランダムに履歴を返す
    def sample(self, num):
        if len(self.state_action_history) < num:
            num = len(self.state_action_history)
        batchs = []
        for state, action in random.sample(self.state_action_history, num):
            batchs.append(
                {
                    "state": state,
                    "action": action,
                    "next_state": self.sample_next_state(state, action),
                    "reward": self.sample_reward(state, action),
                    "done": self.sample_done(state, action),
                    "invalid_actions": [],
                    "next_invalid_actions": [],
                }
            )
        return batchs

    # 次の状態を返す
    def sample_next_state(self, state, action):
        self._init_dict(state, action, None)
        if self.count[state][action] == 0:
            return None
        weights = list(self.trans[state][action].values())
        n_s_list = list(self.trans[state][action].keys())
        n_s = random.choices(n_s_list, weights=weights, k=1)[0]
        return n_s

    # 報酬を返す
    def sample_reward(self, state, action):
        self._init_dict(state, action, None)
        if self.count[state][action] == 0:
            return 0
        return self.reward[state][action] / self.count[state][action]

    # 終了状態を返す
    def sample_done(self, state, action):
        self._init_dict(state, action, None)
        if self.count[state][action] == 0:
            return random.random() < 0.5
        return random.random() < (self.done[state][action] / self.count[state][action])


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.Q = {}
        self.model = _A_MDP()

    def call_restore(self, data: Any, **kwargs) -> None:
        d = pickle.loads(data)
        self.Q = d[0]
        self.model.restore(d[1])

    def call_backup(self, **kwargs):
        return pickle.dumps([self.Q, self.model.backup()])

    def get_action_values(self, state, invalid_actions):
        if state not in self.Q:
            self.Q[state] = [-np.inf if a in invalid_actions else 0 for a in range(self.config.action_space.n)]
        return self.Q[state]


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_sch = SchedulerConfig.create_scheduler(self.config.lr)

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        batchs = self.memory.sample()

        # --- 近似モデルの学習
        model = self.parameter.model
        for batch in batchs:
            # データ形式を変形
            state = batch["state"]
            n_state = batch["next_state"]
            action = batch["action"]
            reward = batch["reward"]
            done = batch["done"]
            invalid_actions = batch["invalid_actions"]
            next_invalid_actions = batch["next_invalid_actions"]

            model.train(state, action, n_state, reward, done)
            self.train_count += 1

        td_error = 0
        lr = self.lr_sch.get_and_update_rate(self.train_count)

        # --- 近似モデルからランダムにサンプリング
        for batch in model.sample(len(batchs) * 2):
            # データ形式を変形
            s = batch["state"]
            n_s = batch["next_state"]
            action = batch["action"]
            reward = batch["reward"]
            done = batch["done"]
            invalid_actions = batch["invalid_actions"]
            next_invalid_actions = batch["next_invalid_actions"]

            q = self.parameter.get_action_values(s, invalid_actions)
            n_q = self.parameter.get_action_values(n_s, next_invalid_actions)

            if done:
                target_q = reward
            else:
                target_q = reward + self.config.discount * max(n_q)

            td_error = target_q - q[action]
            q[action] += lr * td_error

            td_error += td_error

        if len(batchs) > 0:
            td_error /= len(batchs)

        self.info = {
            "size": len(self.parameter.Q),
            "td_error": td_error,
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.epsilon_sch = SchedulerConfig.create_scheduler(self.config.epsilon)

    def on_reset(self, worker) -> InfoType:
        return {}

    def policy(self, worker) -> Tuple[int, InfoType]:
        self.state = self.config.observation_space.to_str(worker.state)
        self.invalid_actions = worker.get_invalid_actions()

        if self.training:
            epsilon = self.epsilon_sch.get_and_update_rate(self.total_step)
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            self.action = random.choice(
                [a for a in range(self.config.action_space.n) if a not in self.invalid_actions]
            )
        else:
            q = self.parameter.get_action_values(self.state, self.invalid_actions)
            q = np.asarray(q)
            q = [(-np.inf if a in self.invalid_actions else v) for a, v in enumerate(q)]
            self.action = np.random.choice(np.where(q == np.max(q))[0])

        return int(self.action), {}

    def on_step(self, worker) -> InfoType:
        if not self.training:
            return {}

        batch = {
            "state": self.state,
            "next_state": self.config.observation_space.to_str(worker.state),
            "action": self.action,
            "reward": worker.reward,
            "done": worker.terminated,
            "invalid_actions": self.invalid_actions,
            "next_invalid_actions": worker.get_invalid_actions(),
        }
        self.memory.add(batch)
        return {}

    def render_terminal(self, worker, **kwargs) -> None:
        q = self.parameter.get_action_values(self.state, self.invalid_actions)
        model = self.parameter.model
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            if a == maxa:
                s = "*"
            else:
                s = " "
            s += f"{worker.env.action_to_str(a)}: Q {q[a]:7.5f}"
            s += f", n_s{str(model.sample_next_state(self.state, a)):10s}"
            s += f", reward {model.sample_reward(self.state, a):.3f}"
            s += f", done {model.sample_done(self.state, a)}"
            return s

        funcs.render_discrete_action(int(maxa), self.config.action_space, worker.env, _render_sub)
