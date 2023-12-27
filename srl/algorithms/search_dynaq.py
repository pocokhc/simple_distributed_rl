import json
import logging
import math
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


def symlog(x):
    return np.sign(x) * np.log(np.abs(x) + 1.0)


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig):
    #:
    q_discount: float = 0.99
    value_iteration_threshold: float = 0.0001
    q_model_update_interval: int = 1000
    q_beta: float = 1.0

    #:
    int_lr: float = 0.1  # type: ignore , type OK
    int_discount: float = 0.9
    int_target_model_update_interval: int = 100

    action_ucb_beta: float = math.sqrt(2)
    episodic_memory_capacity: int = 30000
    lifelong_decrement_rate: float = 0.99  # 減少割合

    def __post_init__(self):
        super().__post_init__()

        self.int_lr: SchedulerConfig = SchedulerConfig(cast(float, self.int_lr))

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
        self.discount = config.q_discount
        self.threshold = config.value_iteration_threshold

        self.trans = {}  # [state][action][next_state] 訪問回数
        self.reward = {}  # [state][action][next_state] 報酬の合計
        self.done = {}  # [state][action][next_state] 終了回数
        self.count = {}  # [state][action][next_state] 訪問回数

    def _backup(self):
        return [
            self.trans,
            self.reward,
            self.done,
            self.count,
        ]

    def _restore(self, d):
        self.trans = d[0]
        self.reward = d[1]
        self.done = d[2]
        self.count = d[3]

    def _init_state(self, state, action, n_state):
        if state not in self.count:
            self.trans[state] = [{} for _ in range(self.action_num)]
            self.reward[state] = [{} for _ in range(self.action_num)]
            self.done[state] = [{} for _ in range(self.action_num)]
            self.count[state] = [{} for _ in range(self.action_num)]
        if n_state is not None and n_state not in self.trans[state][action]:
            self.trans[state][action][n_state] = 0
            self.reward[state][action][n_state] = 0
            self.done[state][action][n_state] = 0
            self.count[state][action][n_state] = 0

    def update(self, state, action, n_state, reward, done):
        self._init_state(state, action, n_state)
        self.trans[state][action][n_state] += 1
        self.reward[state][action][n_state] += reward
        self.done[state][action][n_state] += 1 if done else 0
        self.count[state][action][n_state] += 1

    def compute_value_iteration(self):
        states = self.trans.keys()
        V = {s: 0 for s in states}

        for i in range(50):  # for safety
            delta = 0
            for state in states:
                expected_reward = []
                for act in range(self.action_num):
                    r = 0
                    N = sum(self.trans[state][act].values())
                    if N == 0:
                        continue
                    for next_state in self.trans[state][act].keys():
                        if self.count[state][act][next_state] == 0:
                            continue
                        prob = self.trans[state][act][next_state] / N
                        reward = self.reward[state][act][next_state] / self.count[state][act][next_state]
                        done = self.done[state][act][next_state] / self.count[state][act][next_state]
                        if done > 0.5:
                            gain = reward
                        elif next_state not in V:
                            V[next_state] = 0
                            gain = reward
                        else:
                            gain = reward + self.discount * V[next_state]
                        r += prob * gain
                    expected_reward.append(r)
                # greedy
                maxq = max(expected_reward)
                delta = max(delta, abs(V[state] - maxq))
                V[state] = maxq

            if delta < self.threshold:
                break
        return V

    def compute_action_value_iteration(self):
        V = self.compute_value_iteration()
        Q = {}

        for state in self.trans.keys():
            Q[state] = [0 for _ in range(self.action_num)]
            for act in range(self.action_num):
                r = 0
                N = sum(self.trans[state][act].values())
                if N == 0:
                    continue
                for next_state in self.trans[state][act].keys():
                    if self.count[state][act][next_state] == 0:
                        continue
                    prob = self.trans[state][act][next_state] / N
                    reward = self.reward[state][act][next_state] / self.count[state][act][next_state]
                    done = self.done[state][act][next_state] / self.count[state][act][next_state]
                    if done > 0.5:
                        gain = reward
                    else:
                        gain = reward + self.discount * V[next_state]
                    r += prob * gain
                Q[state][act] = r
        return Q, V


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.Q_int = {}
        self.Q_int_target = {}
        self.lifelong_C = {}
        self.C = {}
        self.model = _A_MDP(self.config)
        self.Q, _ = self.model.compute_action_value_iteration()

    def call_restore(self, data: Any, **kwargs) -> None:
        d = json.loads(data)
        self.Q_int = d[0]
        self.Q_int_target = d[0]
        self.lifelong_C = d[1]
        self.C = d[2]
        self.model._restore(d[3])
        self.Q, _ = self.model.compute_action_value_iteration()

    def call_backup(self, **kwargs):
        return json.dumps(
            [
                self.Q_int,
                self.lifelong_C,
                self.C,
                self.model._backup(),
            ]
        )

    def init_state(self, state: str):
        if state not in self.Q_int:
            self.Q_int[state] = [0.0 for a in range(self.config.action_num)]
            self.Q_int_target[state] = [0.0 for a in range(self.config.action_num)]
            self.C[state] = [0 for a in range(self.config.action_num)]
            self.lifelong_C[state] = 1
        if state not in self.Q:
            self.Q[state] = [0 for a in range(self.config.action_num)]

    def policy(self, state: str, invalid_actions: List[int], q_beta, update: bool):
        self.init_state(state)
        q_ext = symlog(np.asarray(self.Q[state]))
        q = q_beta * q_ext + np.asarray(self.Q_int[state])

        # actionはUCBで決める
        N = sum(self.C[state])
        ucb_list = []
        for i in range(self.config.action_num):
            n = self.C[state][i]
            if n == 0:
                ucb_list.append(np.inf)
            else:
                ucb = q[i] + self.config.action_ucb_beta * np.sqrt(np.log(N) / n)
                ucb_list.append(ucb)

        action = get_random_max_index(ucb_list, invalid_actions)

        if update:
            self.C[state][action] += 1
        return action, ucb_list


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.int_lr_sch = self.config.int_lr.create_schedulers()

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        batchs = self.memory.sample(self.batch_size, self.train_count)

        int_lr = self.int_lr_sch.get_and_update_rate(self.train_count)
        td_error_int = 0
        for batch in batchs:
            state = batch["state"]
            n_state = batch["next_state"]
            action = batch["action"]
            reward_ext = batch["reward_ext"]
            done = batch["done"]
            next_invalid_actions = batch["next_invalid_actions"]
            self.parameter.init_state(state)
            self.parameter.init_state(n_state)

            # --- 近似モデルの学習
            self.parameter.model.update(
                state,
                action,
                n_state,
                reward_ext,
                done,
            )

            # --- int Q
            target_q = batch["reward_int"]
            if not done:
                # on policy
                n_q_idx, _ = self.parameter.policy(n_state, next_invalid_actions, self.config.q_beta, update=False)
                target_q += self.config.int_discount * self.parameter.Q_int_target[n_state][n_q_idx]
            td_error_int = target_q - self.parameter.Q_int[state][action]
            self.parameter.Q_int[state][action] += int_lr * td_error_int

            # --- lifelong
            self.parameter.lifelong_C[state] *= self.config.lifelong_decrement_rate

            self.train_count += 1

        if self.train_count % self.config.int_target_model_update_interval == 0:
            self.parameter.Q_int_target = json.loads(json.dumps(self.parameter.Q_int))

        if self.train_count % self.config.q_model_update_interval == 0:
            self.parameter.Q, _ = self.parameter.model.compute_action_value_iteration()

        self.train_info = {
            "size": len(self.parameter.Q_int),
            "td_error_int": td_error_int,
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        self.episodic_C = {}
        return {}

    def call_policy(self, _state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        self.invalid_actions = invalid_actions
        self.state = to_str_observation(_state)
        self.parameter.init_state(self.state)

        if self.training:
            self.action, _ = self.parameter.policy(self.state, invalid_actions, self.config.q_beta, update=True)
        else:
            self.action = get_random_max_index(self.parameter.Q[self.state], invalid_actions)

        return self.action, {}

    def call_on_step(
        self,
        _next_state: np.ndarray,
        reward_ext: float,
        done: bool,
        next_invalid_actions: List[int],
    ) -> dict:
        if not self.training:
            return {}
        next_state = to_str_observation(_next_state)
        self.parameter.init_state(next_state)

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
            "next_invalid_actions": next_invalid_actions,
        }
        self.memory.add(batch)
        return {}

    def _calc_episodic_reward(self, state, update: bool = True):
        if state not in self.episodic_C:
            self.episodic_C[state] = 0

        # 0回だと無限大になるので1回は保証する
        reward = 1 / math.sqrt(self.episodic_C[state] + 1)

        # 数える
        if update:
            self.episodic_C[state] += 1
        return reward

    def _calc_lifelong_reward(self, state):
        return self.parameter.lifelong_C[state]

    def render_terminal(self, worker: WorkerRun, **kwargs) -> None:
        self.parameter.init_state(self.state)
        _, ucb_list = self.parameter.policy(self.state, worker.get_invalid_actions(), self.config.q_beta, update=False)

        episodic_reward = self._calc_episodic_reward(self.state, update=False)
        lifelong_reward = self._calc_lifelong_reward(self.state)
        int_reward = episodic_reward * lifelong_reward
        print(f"int_reward {int_reward:.4f} = episodic {episodic_reward:.3f} * lifelong {lifelong_reward:.3f}")

        q_ext = self.parameter.Q[self.state]
        q_int = np.asarray(self.parameter.Q_int[self.state])
        maxa = np.argmax(q_ext)

        def _render_sub(a: int) -> str:
            s = f"{q_ext[a]:8.5f}(ext), {q_int[a]:8.5f}(int), {self.parameter.C[self.state][a]}n"
            s += f", ucb {ucb_list[a]:.3f}"
            return s

        render_discrete_action(maxa, worker.env, self.config, _render_sub)
