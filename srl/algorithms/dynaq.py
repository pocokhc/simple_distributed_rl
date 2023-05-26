import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, cast

import numpy as np

from srl.base.define import RLTypes
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.registration import register
from srl.base.rl.remote_memory.sequence_memory import SequenceRemoteMemory
from srl.rl.functions.common import render_discrete_action, to_str_observation

logger = logging.getLogger(__name__)


"""
Other
    invalid_actions : o
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):
    epsilon: float = 0.1
    test_epsilon: float = 0
    discount: float = 0.9
    lr: float = 0.1

    @property
    def observation_type(self) -> RLTypes:
        return RLTypes.DISCRETE

    def getName(self) -> str:
        return "Dyna-Q"


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
        self.config = cast(Config, self.config)

        self.Q = {}
        self.model = _A_MDP()

    def call_restore(self, data: Any, **kwargs) -> None:
        self.Q = json.loads(data)
        # model TODO

    def call_backup(self, **kwargs):
        return json.dumps(self.Q)

    def get_action_values(self, state, invalid_actions):
        if state not in self.Q:
            self.Q[state] = [-np.inf if a in invalid_actions else 0 for a in range(self.config.action_num)]
        return self.Q[state]


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
        # --- 近似モデルの学習
        model = self.parameter.model
        batchs = self.remote_memory.sample()
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

        td_error = 0

        # --- 近似モデルからランダムにサンプリング
        for batch in model.sample(10):
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
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        self.state = to_str_observation(state)
        self.invalid_actions = invalid_actions

        if self.training:
            self.epsilon = self.config.epsilon
        else:
            self.epsilon = self.config.test_epsilon

        return {}

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        self.state = to_str_observation(state)
        self.invalid_actions = invalid_actions

        if random.random() < self.epsilon:
            self.action = cast(int, self.sample_action())
        else:
            q = self.parameter.get_action_values(self.state, invalid_actions)
            q = np.asarray(q)
            q = [(-np.inf if a in invalid_actions else v) for a, v in enumerate(q)]
            self.action = np.random.choice(np.where(q == np.max(q))[0])

        return self.action, {}

    def call_on_step(
        self,
        next_state: Any,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ) -> Dict:
        if not self.training:
            return {}

        batch = {
            "state": self.state,
            "next_state": to_str_observation(next_state),
            "action": self.action,
            "reward": reward,
            "done": done,
            "invalid_actions": self.invalid_actions,
            "next_invalid_actions": next_invalid_actions,
        }
        self.remote_memory.add(batch)
        return {}

    def render_terminal(self, env, worker, **kwargs) -> None:
        q = self.parameter.get_action_values(self.state, self.invalid_actions)
        model = self.parameter.model
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            if a == maxa:
                s = "*"
            else:
                s = " "
            s += f"{env.action_to_str(a)}: Q {q[a]:7.5f}"
            s += f", n_s{model.sample_next_state(self.state, a)}"
            s += f", reward {model.sample_reward(self.state, a):.3f}"
            s += f", done {model.sample_done(self.state, a)}"
            return s

        render_discrete_action(self.invalid_actions, maxa, env, _render_sub)
