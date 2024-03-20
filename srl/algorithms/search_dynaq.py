import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from srl.base.define import DoneTypes, InfoType, RLBaseTypes
from srl.base.exception import UndefinedError
from srl.base.rl.config import RLConfig
from srl.base.rl.parameter import RLParameter
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker import RLWorker
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.rl.functions import helper
from srl.rl.memories.sequence_memory import SequenceMemory

logger = logging.getLogger(__name__)


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig[DiscreteSpace, ArrayDiscreteSpace]):
    #: 学習時の探索率
    search_rate: float = 0.5
    #: テスト時の探索率
    test_search_rate: float = 0.01

    #: アクション選択におけるUCBのペナルティ項の反映率
    action_ucb_penalty_rate: float = 0.1

    #: 近似モデルの学習時に内部報酬を割り引く率
    int_reward_discount: float = 0.9
    #: 方策反復法におけるタイムアウト
    iteration_timeout: float = 1
    #: 方策反復法の学習完了の閾値
    iteration_threshold: float = 0.001
    #: 方策反復法を実行する間隔(学習回数)
    iteration_interval: int = 1_000

    #: 外部報酬の割引率
    q_ext_discount: float = 0.9
    #: 内部報酬の割引率
    q_int_discount: float = 0.9
    #: 外部報酬の学習率
    q_ext_lr: float = 0.1
    #: 内部報酬の学習率
    q_int_lr: float = 0.1
    #: 外部報酬の目標方策で、最大価値を選ぶ確率
    q_ext_target_policy_prob: float = 1.0
    #: 内部報酬の目標方策で、最大価値を選ぶ確率
    q_int_target_policy_prob: float = 0.9

    #: lifelong rewardの減少率
    lifelong_decrement_rate: float = 0.999

    def get_base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    def get_framework(self) -> str:
        return ""

    def get_name(self) -> str:
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


class Memory(SequenceMemory):
    pass


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter[Config]):
    def __init__(self, *args):
        super().__init__(*args)

        # [state][action][next_state]
        self.trans = {}
        self.reward_ext = {}
        self.reward_int = {}
        self.done = {}

        # [state]
        self.invalid_actions = {}

        # [state][action]
        self.q_ext = {}
        self.q_int = {}
        self.action_count = {}

        self.q_ext_min = np.inf
        self.q_ext_max = -np.inf
        self.q_int_min = np.inf
        self.q_int_max = -np.inf

        # [state]
        self.lifelong = {}

    def call_restore(self, data: Any, **kwargs) -> None:
        d = json.loads(data)
        self.trans = d[0]
        self.reward_ext = d[1]
        self.reward_int = d[2]
        self.done = d[3]
        self.invalid_actions = d[4]
        self.q_ext = d[5]
        self.q_ext_min = d[6]
        self.q_ext_max = d[7]
        self.q_int = d[8]
        self.q_int_min = d[9]
        self.q_int_max = d[10]
        self.action_count = d[11]
        self.lifelong = d[12]

    def call_backup(self, **kwargs):
        return json.dumps(
            [
                self.trans,
                self.reward_ext,
                self.reward_int,
                self.done,
                self.invalid_actions,
                self.q_ext,
                self.q_ext_min,
                self.q_ext_max,
                self.q_int,
                self.q_int_min,
                self.q_int_max,
                self.action_count,
                self.lifelong,
            ]
        )

    def init_state(self, state, action, n_state, invalid_actions, next_invalid_actions):
        if state not in self.trans:
            n = self.config.action_space.n
            self.trans[state] = [{} for _ in range(n)]
            self.reward_ext[state] = [{} for _ in range(n)]
            self.reward_int[state] = [{} for _ in range(n)]
            self.done[state] = [{} for _ in range(n)]
            self.invalid_actions[state] = invalid_actions
        if n_state is not None and n_state not in self.trans[state][action]:
            self.trans[state][action][n_state] = 0
            self.reward_ext[state][action][n_state] = 0.0
            self.reward_int[state][action][n_state] = 1.0
            self.done[state][action][n_state] = 0.0
            self.invalid_actions[n_state] = next_invalid_actions

    def init_q(self, state: str):
        if state not in self.q_ext:
            n = self.config.action_space.n
            self.q_ext[state] = [0.0 for a in range(n)]
            self.q_int[state] = [0.0 for a in range(n)]
            self.action_count[state] = [0 for a in range(n)]
            self.lifelong[state] = 1.0

    def iteration_q(
        self,
        mode: str,
        threshold: float = 0.0001,
        timeout: float = 1,
    ):
        if mode == "ext":
            discount = self.config.q_ext_discount
            q_tbl = self.q_ext
            reward_tbl = self.reward_ext
            prob = self.config.q_ext_target_policy_prob
            self.q_ext_min = np.inf
            self.q_ext_max = -np.inf
        elif mode == "int":
            discount = self.config.q_int_discount
            q_tbl = self.q_int
            reward_tbl = self.reward_int
            prob = self.config.q_int_target_policy_prob
            self.q_int_min = np.inf
            self.q_int_max = -np.inf
        else:
            raise UndefinedError(mode)

        delta = 0
        t0 = time.time()
        while time.time() - t0 < timeout:  # for safety
            delta = 0
            for state in self.trans.keys():
                for act in range(self.config.action_space.n):
                    if act in self.invalid_actions[state]:
                        continue
                    N = sum(self.trans[state][act].values())
                    if N == 0:
                        continue

                    q = 0
                    for next_state in self.trans[state][act].keys():
                        if self.trans[state][act][next_state] == 0:
                            continue
                        trans_prob = self.trans[state][act][next_state] / N
                        reward = reward_tbl[state][act][next_state]
                        done = self.done[state][act][next_state]
                        n_q = self.calc_next_q(q_tbl[next_state], prob, self.invalid_actions[next_state])
                        gain = reward + (1 - done) * discount * n_q
                        q += trans_prob * gain

                    delta = max(delta, abs(q_tbl[state][act] - q))
                    q_tbl[state][act] = q

            if delta < threshold:
                break
        else:
            logger.info(f"[{mode}] iteration timeout(delta={delta}, threshold={threshold})")

        # update range
        for state in self.trans.keys():
            for act in range(self.config.action_space.n):
                if act in self.invalid_actions[state]:
                    continue
                if mode == "ext":
                    if self.q_ext_min > self.q_ext[state][act]:
                        self.q_ext_min = self.q_ext[state][act]
                    if self.q_ext_max < self.q_ext[state][act]:
                        self.q_ext_max = self.q_ext[state][act]
                elif mode == "int":
                    if self.q_int_min > self.q_int[state][act]:
                        self.q_int_min = self.q_int[state][act]
                    if self.q_int_max < self.q_int[state][act]:
                        self.q_int_max = self.q_int[state][act]

    def calc_next_q(self, q_tbl, prob: float, invalid_actions):
        if self.config.action_space.n == len(invalid_actions):
            # 有効アクションがない場合
            return 0

        q_max = max(q_tbl)
        if prob == 1:
            return q_max

        q_max_idx = [a for a, q in enumerate(q_tbl) if q == q_max and (a not in invalid_actions)]
        valid_actions = self.config.action_space.n - len(invalid_actions)
        if valid_actions == len(q_max_idx):
            prob = 1.0

        n_q = 0
        for a in range(self.config.action_space.n):
            if a in invalid_actions:
                continue
            elif a in q_max_idx:
                p = prob / len(q_max_idx)
            else:
                p = (1 - prob) / (valid_actions - len(q_max_idx))
            n_q += p * q_tbl[a]
        return n_q

    def calc_q_normalize(self, q: np.ndarray, q_min: float, q_max: float):
        if q_min >= q_max:
            return q
        return (q - q_min) / (q_max - q_min)

    def sample_next_state(self, state: str, action: int):
        if state not in self.trans:
            return None
        n_s_list = list(self.trans[state][action].keys())
        if len(n_s_list) == 0:
            return None
        weights = list(self.trans[state][action].values())
        r_idx = helper.random_choice_by_probs(weights)
        return n_s_list[r_idx]


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.iteration_num = 0

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        batchs = self.memory.sample()

        td_error_ext = 0
        td_error_int = 0
        for batch in batchs:
            state = batch["state"]
            n_state = batch["next_state"]
            action = batch["action"]
            reward_ext = batch["reward_ext"]
            reward_int = batch["reward_int"]
            done = batch["done"]
            invalid_actions = batch["invalid_actions"]
            next_invalid_actions = batch["next_invalid_actions"]

            self.parameter.init_state(state, action, n_state, invalid_actions, next_invalid_actions)
            self.parameter.init_q(state)
            self.parameter.init_q(n_state)

            done = 1 if done else 0

            # --- model update
            self.parameter.trans[state][action][n_state] += 1
            c = self.parameter.trans[state][action][n_state]
            # online mean
            self.parameter.done[state][action][n_state] += (done - self.parameter.done[state][action][n_state]) / c
            # online mean
            self.parameter.reward_ext[state][action][n_state] += (
                reward_ext - self.parameter.reward_ext[state][action][n_state]
            ) / c
            # discount online mean
            old_reward_int = self.parameter.reward_int[state][action][n_state]
            self.parameter.reward_int[state][action][n_state] = (
                old_reward_int * self.config.int_reward_discount
                + (reward_int - old_reward_int * self.config.int_reward_discount) / c
            )

            # --- ext (greedy)
            # TD誤差なので生データ
            n_q = self.parameter.calc_next_q(
                self.parameter.q_ext[n_state], self.config.q_ext_target_policy_prob, next_invalid_actions
            )
            target_q = reward_ext + (1 - done) * self.config.q_ext_discount * n_q
            td_error_ext = target_q - self.parameter.q_ext[state][action]
            self.parameter.q_ext[state][action] += self.config.q_ext_lr * td_error_ext

            if self.parameter.q_ext_min > self.parameter.q_ext[state][action]:
                self.parameter.q_ext_min = self.parameter.q_ext[state][action]
            if self.parameter.q_ext_max < self.parameter.q_ext[state][action]:
                self.parameter.q_ext_max = self.parameter.q_ext[state][action]

            # --- int (sarsa)
            n_q = self.parameter.calc_next_q(
                self.parameter.q_int[n_state], self.config.q_int_target_policy_prob, next_invalid_actions
            )
            target_q = reward_int + (1 - done) * self.config.q_int_discount * n_q
            td_error_int = target_q - self.parameter.q_int[state][action]
            self.parameter.q_int[state][action] += self.config.q_int_lr * td_error_int

            if self.parameter.q_int_min > self.parameter.q_int[state][action]:
                self.parameter.q_int_min = self.parameter.q_int[state][action]
            if self.parameter.q_int_max < self.parameter.q_int[state][action]:
                self.parameter.q_int_max = self.parameter.q_int[state][action]

            # --- lifelong
            self.parameter.lifelong[state] *= self.config.lifelong_decrement_rate

            # --- action count update
            if self.distributed:
                self.parameter.action_count[state][action] += 1

            if self.train_count % self.config.iteration_interval == 0:
                self.parameter.iteration_q("ext", self.config.iteration_threshold, self.config.iteration_timeout)
                self.parameter.iteration_q("int", self.config.iteration_threshold, self.config.iteration_timeout)
                self.iteration_num += 1

            self.train_count += 1

        self.train_info = {
            "size": len(self.parameter.q_ext),
            "td_error_ext": abs(td_error_ext),
            "td_error_int": abs(td_error_int),
            "iteration": self.iteration_num,
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker[Config, Parameter, DiscreteSpace, ArrayDiscreteSpace]):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

    def on_start(self, worker) -> None:
        self.parameter.iteration_q("ext", self.config.iteration_threshold / 10, self.config.iteration_timeout * 2)
        self.parameter.iteration_q("int", self.config.iteration_threshold, self.config.iteration_timeout)

    def on_reset(self, worker) -> InfoType:
        self.episodic = {}
        return {}

    def policy(self, worker) -> Tuple[int, InfoType]:
        self.state = self.observation_space.to_str(worker.state)
        invalid_actions = worker.invalid_actions
        self.parameter.init_q(self.state)

        q_ext = np.array(self.parameter.q_ext[self.state])
        q_int = np.array(self.parameter.q_int[self.state])
        q_ext = self.parameter.calc_q_normalize(q_ext, self.parameter.q_ext_min, self.parameter.q_ext_max)
        q_int = self.parameter.calc_q_normalize(q_int, self.parameter.q_int_min, self.parameter.q_int_max)

        if self.training or self.rendering:
            # 外部報酬が疎な場合は探索を優先
            if self.parameter.q_ext_min >= self.parameter.q_ext_max:
                q = q_int
            else:
                q = (1 - self.config.search_rate) * q_ext + self.config.search_rate * q_int

            # actionはUCBで決める
            N = sum(self.parameter.action_count[self.state])
            self.ucb_list = []
            for a in range(self.action_space.n):
                if a in invalid_actions:
                    self.ucb_list.append(-np.inf)
                    continue
                n = self.parameter.action_count[self.state][a]
                if n == 0:
                    self.ucb_list.append(np.inf)
                else:
                    ucb = q[a] + self.config.action_ucb_penalty_rate * np.sqrt(2 * np.log(N) / n)
                    self.ucb_list.append(ucb)

        if self.training:
            self.action = helper.get_random_max_index(self.ucb_list, invalid_actions)
            self.parameter.action_count[self.state][self.action] += 1
        else:
            q = (1 - self.config.test_search_rate) * q_ext + self.config.test_search_rate * q_int
            self.action = helper.get_random_max_index(q, invalid_actions)

        return self.action, {}

    def on_step(self, worker) -> InfoType:
        if not self.training:
            return {}
        next_state = self.observation_space.to_str(worker.state)
        self.parameter.init_q(next_state)

        # 内部報酬
        episodic_reward = self._calc_episodic_reward(next_state)
        lifelong_reward = self._calc_lifelong_reward(next_state)
        reward_int = episodic_reward * lifelong_reward

        batch = {
            "state": self.state,
            "next_state": next_state,
            "action": self.action,
            "reward_ext": worker.reward,
            "reward_int": reward_int,
            "done": self.worker.done_type == DoneTypes.TERMINATED,
            "invalid_actions": worker.prev_invalid_actions,
            "next_invalid_actions": worker.invalid_actions,
        }
        self.memory.add(batch)
        return {}

    def _calc_episodic_reward(self, state, update: bool = True):
        if state not in self.episodic:
            self.episodic[state] = 0

        # 0回だと無限大になるので1回は保証する
        reward = 1 / math.sqrt(self.episodic[state] + 1)

        # 数える
        if update:
            self.episodic[state] += 1
        return reward

    def _calc_lifelong_reward(self, state):
        return self.parameter.lifelong[state]

    def render_terminal(self, worker, **kwargs) -> None:
        prev_state = self.observation_space.to_str(worker.prev_state)
        act = worker.prev_action
        state = self.observation_space.to_str(worker.state)
        self.parameter.init_state(prev_state, act, state, worker.prev_invalid_actions, worker.invalid_actions)
        self.parameter.init_q(prev_state)
        self.parameter.init_q(state)

        r_ext = self.parameter.reward_ext[prev_state][act][state]
        r_int = self.parameter.reward_int[prev_state][act][state]
        done = self.parameter.done[prev_state][act][state]
        s = f"reward_ext {r_ext:8.5f}"
        s += f", reward_int {r_int:8.5f}"
        s += f", done {done:.1%}"
        print(s)

        episodic_reward = self._calc_episodic_reward(state, update=False)
        lifelong_reward = self._calc_lifelong_reward(state)
        int_reward = episodic_reward * lifelong_reward
        print(f"int_reward {int_reward:.4f} = episodic {episodic_reward:.3f} * lifelong {lifelong_reward:.3f}")

        q_ext = np.array(self.parameter.q_ext[state])
        q_int = np.array(self.parameter.q_int[state])
        q_ext_nor = self.parameter.calc_q_normalize(q_ext, self.parameter.q_ext_min, self.parameter.q_ext_max)
        q_int_nor = self.parameter.calc_q_normalize(q_int, self.parameter.q_int_min, self.parameter.q_int_max)
        q = (1 - self.config.test_search_rate) * q_ext_nor + self.config.test_search_rate * q_int_nor
        maxa = np.argmax(q)
        print(f"q_ext range[{self.parameter.q_ext_min:.3f}, {self.parameter.q_ext_max:.3f}]")
        print(f"q_int range[{self.parameter.q_int_min:.3f}, {self.parameter.q_int_max:.3f}]")

        def _render_sub(a: int) -> str:
            s = f"{q_ext[a]:7.2f}->{q_ext_nor[a]:6.3f}(ext), {q_int[a]:6.3f}->{q_int_nor[a]:6.3f}(int)"
            s += f", {self.parameter.action_count[self.state][a]:4d}n"
            s += f", ucb {self.ucb_list[a]:.3f}"
            return s

        helper.render_discrete_action(int(maxa), self.config.action_space.n, worker.env, _render_sub)
