import json
import logging
import random
from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import numpy as np

from srl.base.define import RLBaseTypes
from srl.base.rl.algorithms.discrete_action import DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.config import RLConfig
from srl.base.rl.registration import register
from srl.base.rl.worker_run import WorkerRun
from srl.rl.functions.common import get_random_max_index, render_discrete_action, to_str_observation
from srl.rl.memories.sequence_memory import SequenceMemory
from srl.rl.schedulers.scheduler import SchedulerConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig):
    test_ext_beta: float = 1.0
    test_int_beta: float = 0.0

    search_mode: bool = True
    actor_num: int = 5
    actor_ucb_discount = 0.9
    actor_ucb_beta: float = np.sqrt(2)
    action_ucb_beta: float = np.sqrt(2)

    # model params
    lr_ext: float = 0.1  # type: ignore , type OK
    ext_discount: float = 0.9
    lr_int: float = 0.5  # type: ignore , type OK
    int_discount: float = 0.1

    # episodic
    episodic_memory_capacity: int = 30000

    # lifelong
    lifelong_decrement_rate: float = 0.99  # 減少割合
    lifelong_reward_L: float = 5.0

    # other
    q_init: str = ""

    def __post_init__(self):
        super().__post_init__()

        self.lr_ext: SchedulerConfig = SchedulerConfig(cast(float, self.lr_ext))
        self.lr_int: SchedulerConfig = SchedulerConfig(cast(float, self.lr_int))

    @property
    def base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    @property
    def base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    def get_use_framework(self) -> str:
        return ""

    def getName(self) -> str:
        return "SearchDynaQ"

    def assert_params(self) -> None:
        super().assert_params()


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
    def __init__(self, config: Config) -> None:
        self.action_num = config.action_num

        self.trans = {}  # [state][action][next_state] 訪問回数
        self.reward = {}  # [state][action] 報酬の合計
        self.done = {}  # [state][action] 終了回数
        self.count = {}  # [state][action] 訪問回数

        # 訪問履歴(state,action)
        self.existing_location = []

    def _backup(self):
        return [
            self.trans,
            self.reward,
            self.done,
            self.count,
            self.existing_location,
        ]

    def _restore(self, d):
        self.trans = d[0]
        self.reward = d[1]
        self.done = d[2]
        self.count = d[3]
        self.existing_location = d[4]

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

    def update(self, state, action, n_state, reward, done):
        self._init_state(state, action, n_state)
        self.trans[state][action][n_state] += 1
        self.reward[state][action] += reward
        self.done[state][action] += 1 if done else 0
        self.count[state][action] += 1

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
        self.config: Config = self.config

        self.Q_ext = {}
        self.Q_int = {}
        self.lifelong_C = {}
        self.C = {}
        self.model = _A_MDP(self.config)

    def call_restore(self, data: Any, **kwargs) -> None:
        d = json.loads(data)
        self.Q_ext = d[0]
        self.Q_int = d[1]
        self.lifelong_C = d[2]
        self.C = d[3]
        self.model._restore(d[4])

    def call_backup(self, **kwargs):
        return json.dumps(
            [
                self.Q_ext,
                self.Q_int,
                self.lifelong_C,
                self.C,
                self.model._backup(),
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
            self.C[state] = [0 for a in range(self.config.action_num)]


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.ext_lr_sch = self.config.lr_ext.create_schedulers()
        self.int_lr_sch = self.config.lr_int.create_schedulers()

    def train_on_batchs(self, memory_sample_return) -> None:
        batchs = memory_sample_return

        if not self.config.search_mode:
            return

        int_lr = self.int_lr_sch.get_and_update_rate(self.train_count)
        ext_lr = self.ext_lr_sch.get_and_update_rate(self.train_count)
        td_error_int = 0
        td_error_ext = 0
        for batch in batchs:
            state = batch["state"]
            n_state = batch["next_state"]
            action = batch["action"]
            reward_ext = batch["reward_ext"]
            done = batch["done"]
            invalid_actions = batch["invalid_actions"]
            next_invalid_actions = batch["next_invalid_actions"]
            self.parameter.init_state(state, invalid_actions)
            self.parameter.init_state(n_state, next_invalid_actions)

            # --- 近似モデルの学習
            self.parameter.model.update(
                state,
                action,
                n_state,
                reward_ext,
                done,
            )

            # --- int Q
            td_error_int = self._calc_td_error(
                batch["reward_int"],
                done,
                self.parameter.Q_int[state][action],
                self.parameter.Q_int[n_state],
                self.config.int_discount,
            )
            self.parameter.Q_int[state][action] += int_lr * td_error_int

            # --- ext Q
            td_error_ext = self._calc_td_error(
                reward_ext,
                done,
                self.parameter.Q_ext[state][action],
                self.parameter.Q_ext[n_state],
                self.config.ext_discount,
            )
            self.parameter.Q_ext[state][action] += ext_lr * td_error_ext

            self.train_count += 1

        self.train_info = {
            "Q_int": len(self.parameter.Q_int),
            "td_error_int": td_error_int,
            "td_error_ext": td_error_ext,
        }

    def train_no_batchs(self) -> None:
        if self.config.search_mode:
            return

        ext_lr = self.ext_lr_sch.get_and_update_rate(self.train_count)

        batch = self.parameter.model.sample()
        if batch is None:
            return

        state = batch["state"]
        n_state = batch["next_state"]
        action = batch["action"]
        reward = batch["reward"]
        done = batch["done"]

        td_error_ext = self._calc_td_error(
            reward,
            done,
            self.parameter.Q_ext[state][action],
            self.parameter.Q_ext[n_state],
            self.config.ext_discount,
        )
        self.parameter.Q_ext[state][action] += ext_lr * td_error_ext

        self.train_count += 1
        self.train_info = {"td_error_ext": td_error_ext}

    def _calc_td_error(self, reward, done, action_q, n_q, discount):
        target_q = reward
        if not done:
            target_q += discount * max(n_q)
        return target_q - action_q


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.actor_beta_list = np.linspace(0, 1, self.config.actor_num)

        # ucb
        self.actor_index = -1
        self.ucb_all_count = 0
        self.ucb_count = [0 for _ in range(self.config.actor_num)]
        self.ucb_reward = [0.0 for _ in range(self.config.actor_num)]

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        if self.config.search_mode:
            # エピソードの最初に探索率を決める
            self.actor_index = self._calc_actor_index()
            self.actor_beta = self.actor_beta_list[self.actor_index]

        self.prev_episode_reward = 0
        self.episodic_C = {}

        return {}

    def _calc_actor_index(self) -> int:
        # Discounted UCB
        if self.actor_index != -1:
            self.ucb_all_count += 1
            self.ucb_count[self.actor_index] += 1
            n = self.ucb_count[self.actor_index]
            self.ucb_reward[self.actor_index] += (
                self.prev_episode_reward - self.config.actor_ucb_discount * self.ucb_reward[self.actor_index]
            ) / n

        max_ucb = 0
        max_idx = 0
        for i in range(self.config.actor_num):
            if self.ucb_count[i] == 0:
                return i
            n = self.ucb_count[i]
            u = self.ucb_reward[i]
            ucb = u + self.config.actor_ucb_beta * np.sqrt(np.log(self.ucb_all_count) / n)
            if max_ucb < ucb:
                max_ucb = ucb
                max_idx = i
        return max_idx

    def call_policy(self, _state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        self.invalid_actions = invalid_actions
        self.state = to_str_observation(_state)
        self.parameter.init_state(self.state, self.invalid_actions)

        q_ext = np.asarray(self.parameter.Q_ext[self.state])
        q_int = np.asarray(self.parameter.Q_int[self.state])
        if self.config.search_mode:
            q = self.actor_beta * q_ext + q_int

            # actionはUCBで決める
            N = sum(self.parameter.C[self.state])
            self.ucb_list = []
            for i in range(self.config.action_num):
                n = self.parameter.C[self.state][i]
                if n == 0:
                    self.ucb_list.append(np.inf)
                else:
                    ucb = q[i] * self.config.action_ucb_beta * np.sqrt(np.log(N) / n)
                    self.ucb_list.append(ucb)

            self.action = get_random_max_index(self.ucb_list, invalid_actions)
            self.parameter.C[self.state][self.action] += 1  # TODO
            return self.action, {}

        q = self.config.test_ext_beta * q_ext + self.config.test_int_beta * q_int
        self.action = get_random_max_index(q, invalid_actions)
        return self.action, {}

    def call_on_step(
        self,
        _next_state: np.ndarray,
        reward_ext: float,
        done: bool,
        next_invalid_actions: List[int],
    ) -> dict:
        self.prev_episode_reward += reward_ext
        if not self.training:
            return {}
        next_state = to_str_observation(_next_state)
        self.parameter.init_state(next_state, next_invalid_actions)

        # 内部報酬
        episodic_reward = self._calc_episodic_reward(next_state)
        lifelong_reward = self._calc_lifelong_reward(next_state)
        reward_int = episodic_reward * lifelong_reward

        batch = {
            "state": self.state,
            "next_state": next_state,
            "action": self.action,
            "reward_ext": reward_ext,
            "reward_int": reward_int,
            "done": done,
            "invalid_actions": self.invalid_actions,
            "next_invalid_actions": next_invalid_actions,
        }
        self.memory.add(batch)
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

    def render_terminal(self, worker: WorkerRun, **kwargs) -> None:
        self.parameter.init_state(self.state, self.invalid_actions)

        episodic_reward = self._calc_episodic_reward(self.state, update=False)
        lifelong_reward = self._calc_lifelong_reward(self.state, update=False)
        int_reward = episodic_reward * lifelong_reward
        print(f"int_reward {int_reward:.4f} = episodic {episodic_reward:.3f} * lifelong {lifelong_reward:.3f}")

        q_ext = np.asarray(self.parameter.Q_ext[self.state])
        q_int = np.asarray(self.parameter.Q_int[self.state])
        q = self.config.test_ext_beta * q_ext + self.config.test_int_beta * q_int
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            s = f"{q_ext[a]:8.5f}(ext), {q_int[a]:8.5f}(int), {self.parameter.C[self.state][a]}n"
            if self.config.search_mode:
                s += f", {self.ucb_list[a]}"
            return s

        render_discrete_action(maxa, worker.env, self.config, _render_sub)
