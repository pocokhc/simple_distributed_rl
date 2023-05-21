import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, cast

import numpy as np

from srl.base.define import RLObservationTypes
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import SequenceRemoteMemory
from srl.rl.functions.common import render_discrete_action, to_str_observation

logger = logging.getLogger(__name__)


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):
    test_search_rate: float = 0.0
    test_epsilon: float = 0.0

    search_rate: float = 0.9
    epsilon: float = 0.01

    num_q_train: int = 10

    # model params
    ext_lr: float = 0.1
    ext_discount: float = 0.9
    int_lr: float = 0.1
    int_discount: float = 0.9

    # episodic
    episodic_memory_capacity: int = 30000

    # lifelong
    lifelong_decrement_rate: float = 0.999  # 減少割合
    lifelong_reward_L: float = 5.0

    # other
    q_init: str = ""

    @property
    def observation_type(self) -> RLObservationTypes:
        return RLObservationTypes.DISCRETE

    def getName(self) -> str:
        return "SearchDynaQ"

    def assert_params(self) -> None:
        super().assert_params()


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
    def __init__(self, config: Config) -> None:
        self.action_num = config.action_num

        self.trans = {}  # [state][action][next_state] 訪問回数
        self.reward = {}  # [state][action] 報酬の合計
        self.done = {}  # [state][action] 終了回数
        self.count = {}  # [state][action] 訪問回数
        self.all_count = 0

        # 訪問履歴(state,action)
        self.existing_location = []

        self.invalid_actions_list = {}  # [state]

    def _init_state(self, state, action, n_state):
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

    def update(self, state, invalid_actions, action, n_state, n_invalid_actions, reward, done):
        self._init_state(state, action, n_state)
        self.trans[state][action][n_state] += 1
        self.reward[state][action] += reward
        self.done[state][action] += 1 if done else 0
        self.count[state][action] += 1
        self.all_count += 1

        self.invalid_actions_list[state] = invalid_actions
        self.invalid_actions_list[n_state] = n_invalid_actions

        key = (state, action)
        if key not in self.existing_location:
            self.existing_location.append(key)

    # 履歴から架空のbatchを返す
    def sample(self):
        if len(self.existing_location) == 0:
            return None
        state, action = random.choice(self.existing_location)
        next_state = self.sample_next_state(state, action)
        if next_state is None:
            return None
        batch = {
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": self.sample_reward(state, action),
            "done": self.sample_done(state, action),
            "invalid_actions": self.invalid_actions_list[state],
            "next_invalid_actions": self.invalid_actions_list[next_state],
        }
        return batch

    def sample_next_state(self, state, action):
        self._init_state(state, action, None)
        if self.count[state][action] == 0:
            return None
        weights = list(self.trans[state][action].values())
        n_s_list = list(self.trans[state][action].keys())
        n_s = random.choices(n_s_list, weights=weights, k=1)[0]
        return n_s

    def sample_reward(self, state, action):
        self._init_state(state, action, None)
        if self.count[state][action] == 0:
            return 0
        return self.reward[state][action] / self.count[state][action]

    def sample_done(self, state, action, return_prob=False):
        self._init_state(state, action, None)
        if self.count[state][action] == 0:
            prob = 0.5
        else:
            prob = self.done[state][action] / self.count[state][action]
        if return_prob:
            return prob
        else:
            return random.random() < prob


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.Q_ext = {}
        self.Q_int = {}
        self.lifelong_C = {}
        self.Q_C = {}

        self.model = _A_MDP(self.config)

    def call_restore(self, data: Any, **kwargs) -> None:
        d = json.loads(data)
        self.Q_ext = d[0]
        self.Q_int = d[1]
        self.lifelong_C = d[2]
        self.Q_C = d[3]
        # self.model = d[4] TODO

    def call_backup(self, **kwargs):
        return json.dumps(
            [
                self.Q_ext,
                self.Q_int,
                self.lifelong_C,
                self.Q_C,
                # self.model, TODO
            ]
        )

    def init_state(self, state: str, invalid_actions: List[int]):
        if state not in self.Q_ext:
            self.Q_ext[state] = [
                -np.inf if a in invalid_actions else (np.random.normal() if self.config.q_init == "random" else 0.0)
                for a in range(self.config.action_num)
            ]
            L = self.config.lifelong_reward_L
            self.Q_int[state] = [0.0 if a in invalid_actions else L for a in range(self.config.action_num)]
            self.Q_C[state] = 0


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
        if self.remote_memory.length() == 0:
            return {}

        model = self.parameter.model

        # ---------------------
        # 近似モデルの学習
        # ---------------------
        td_error_mean = 0
        batchs = self.remote_memory.sample()
        for batch in batchs:
            state = batch["state"]
            n_state = batch["next_state"]
            action = batch["action"]
            reward_ext = batch["reward_ext"]
            done = batch["done"]
            invalid_actions = batch["invalid_actions"]
            next_invalid_actions = batch["next_invalid_actions"]

            model.update(
                state,
                invalid_actions,
                action,
                n_state,
                next_invalid_actions,
                reward_ext,
                done,
            )

            # --- 内部報酬の学習
            reward_int = batch["reward_int"]

            self.parameter.init_state(state, invalid_actions)
            self.parameter.init_state(n_state, next_invalid_actions)
            q = self.parameter.Q_int[state]
            n_q = self.parameter.Q_int[n_state]

            if done:
                target_q = reward_int
            else:
                maxq = max(n_q)
                # --- original code
                if self.parameter.Q_C[n_state] > 1:
                    maxq = maxq / self.parameter.Q_C[n_state]
                # ---
                target_q = reward_int + self.config.int_discount * maxq

            td_error = target_q - q[action]
            self.parameter.Q_int[state][action] += self.config.int_lr * td_error
            self.parameter.Q_C[state] += 1

            td_error_mean += td_error
        if len(batchs) > 0:
            td_error_mean /= len(batchs)

        _info = {
            "Q_int": len(self.parameter.Q_int),
            "td_error_int": td_error_mean,
        }

        # ---------------------
        # q ext
        # ---------------------
        td_error_mean = 0
        for _ in range(self.config.num_q_train):
            batch = model.sample()
            if batch is None:
                continue

            state = batch["state"]
            n_state = batch["next_state"]
            action = batch["action"]
            reward = batch["reward"]
            done = batch["done"]
            invalid_actions = batch["invalid_actions"]
            next_invalid_actions = batch["next_invalid_actions"]

            self.parameter.init_state(state, invalid_actions)
            self.parameter.init_state(n_state, next_invalid_actions)
            q = self.parameter.Q_ext[state]
            n_q = self.parameter.Q_ext[n_state]

            if done:
                target_q = reward
            else:
                target_q = reward + self.config.ext_discount * max(n_q)

            td_error = target_q - q[action]
            self.parameter.Q_ext[state][action] += self.config.ext_lr * td_error

            td_error_mean += td_error
            self.train_count += 1
        if len(batchs) > 0:
            td_error_mean /= len(batchs)

        _info["td_error_ext"] = td_error_mean
        return _info


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
        self.episodic_C = {}

        self.episodic_reward = 0
        self.lifelong_reward = 0
        self.reward_int = 0

        return {}

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        self.invalid_actions = invalid_actions
        self.state = to_str_observation(state)
        self.parameter.init_state(self.state, self.invalid_actions)

        if self.training:
            epsilon = self.config.epsilon
            search_rate = self.config.search_rate
        else:
            epsilon = self.config.test_epsilon
            search_rate = self.config.test_search_rate

        if random.random() < epsilon:
            action = np.random.choice([a for a in range(self.config.action_num) if a not in invalid_actions])
        else:
            q_ext = np.asarray(self.parameter.Q_ext[self.state])
            q_int = np.asarray(self.parameter.Q_int[self.state])
            q = (1 - search_rate) * q_ext + search_rate * q_int
            q = [(-np.inf if a in invalid_actions else v) for a, v in enumerate(q)]
            action = np.random.choice(np.where(q == np.max(q))[0])

        self.action = int(action)
        return self.action, {}

    def call_on_step(
        self,
        _next_state: np.ndarray,
        reward_ext: float,
        done: bool,
        next_invalid_actions: List[int],
    ) -> Dict:
        if not self.training:
            return {}
        next_state = to_str_observation(_next_state)

        self.episodic_reward = self._calc_episodic_reward(next_state)
        self.lifelong_reward = self._calc_lifelong_reward(next_state)
        self.reward_int = self.episodic_reward * self.lifelong_reward

        batch = {
            "state": self.state,
            "next_state": next_state,
            "action": self.action,
            "reward_ext": reward_ext,
            "reward_int": self.reward_int,
            "done": done,
            "invalid_actions": self.invalid_actions,
            "next_invalid_actions": next_invalid_actions,
        }
        self.remote_memory.add(batch)
        return {}

    def _calc_episodic_reward(self, state, update: bool = True):
        if state not in self.episodic_C:
            self.episodic_C[state] = 0

        # 0回だと無限大になるので1回は保証する
        reward = 1 / np.sqrt(self.episodic_C[state] + 1)

        # 数える
        if update:
            self.episodic_C[state] += 1
        return reward

    def _calc_lifelong_reward(self, state, update: bool = True):
        # RNDと同じ発想で、回数を重ねると0に近づくようにする
        if state not in self.parameter.lifelong_C:
            self.parameter.lifelong_C[state] = self.config.lifelong_reward_L - 1.0  # 初期値
        reward = self.parameter.lifelong_C[state]

        # 0に近づける
        if update:
            self.parameter.lifelong_C[state] *= self.config.lifelong_decrement_rate
        return reward + 1.0

    def render_terminal(self, env, worker, **kwargs) -> None:
        self.parameter.init_state(self.state, self.invalid_actions)

        s = f"int_reward {self.reward_int:.4f} = "
        s += f"episodic {self.episodic_reward:.3f} * lifelong {self.lifelong_reward:.3f}"
        print(s)
        q_ext = np.asarray(self.parameter.Q_ext[self.state])
        q_int = np.asarray(self.parameter.Q_int[self.state])

        q = (1 - self.config.test_search_rate) * q_ext + self.config.test_search_rate * q_int
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            s = f"{q[a]:8.5f} = {q_ext[a]:8.5f}(ext) + {q_int[a]:8.5f}(int) ({self.config.test_search_rate:.2f})"
            return s

        render_discrete_action(self.invalid_actions, maxa, env, _render_sub)
