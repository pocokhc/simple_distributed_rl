import json
import logging
import math
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from srl.base.define import DoneTypes, RLBaseTypes
from srl.base.exception import UndefinedError
from srl.base.rl.algorithms.discrete_action import DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.config import RLConfig
from srl.base.rl.registration import register
from srl.base.rl.worker_run import WorkerRun
from srl.rl.functions.common import (
    get_random_max_index,
    random_choice_by_probs,
    render_discrete_action,
    to_str_observation,
)
from srl.rl.memories.sequence_memory import SequenceMemory

logger = logging.getLogger(__name__)


def symlog(x):
    return np.sign(x) * np.log(np.abs(x) + 1.0)


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig):
    #:
    q_ext_discount: float = 0.9
    q_int_discount: float = 0.9

    value_iteration_threshold: float = 0.0001
    q_model_update_interval: int = 1000

    use_symlog: bool = True

    q_ext_beta: float = 1.0
    q_int_beta: float = 1.0

    q_ext_lr: float = 0.1
    q_int_lr: float = 0.1

    q_ext_target_policy_prob: float = 1.0
    q_int_target_policy_prob: float = 0.9

    int_reward_discount: float = 0.9

    q_max_iteration: int = 10_000
    iteration_threshold: float = 0.01
    iteration_interval: int = 1000

    action_ucb_beta: float = math.sqrt(2)
    episodic_memory_capacity: int = 30000
    lifelong_decrement_rate: float = 0.99  # 減少割合

    def __post_init__(self):
        super().__post_init__()

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
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

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
        self.q_int = d[6]
        self.action_count = d[7]
        self.lifelong = d[8]

    def call_backup(self, **kwargs):
        return json.dumps(
            [
                self.trans,
                self.reward_ext,
                self.reward_int,
                self.done,
                self.invalid_actions,
                self.q_ext,
                self.q_int,
                self.action_count,
                self.lifelong,
            ]
        )

    def init_state(self, state, action, n_state, invalid_actions, next_invalid_actions):
        if state not in self.trans:
            self.trans[state] = [{} for _ in range(self.config.action_num)]
            self.reward_ext[state] = [{} for _ in range(self.config.action_num)]
            self.reward_int[state] = [{} for _ in range(self.config.action_num)]
            self.done[state] = [{} for _ in range(self.config.action_num)]
            self.invalid_actions[state] = invalid_actions
        if n_state is not None and n_state not in self.trans[state][action]:
            self.trans[state][action][n_state] = 0
            self.reward_ext[state][action][n_state] = 0
            self.reward_int[state][action][n_state] = 0
            self.done[state][action][n_state] = 0
            self.invalid_actions[n_state] = next_invalid_actions

    def init_q(self, state: str):
        if state not in self.q_ext:
            self.q_ext[state] = [0.0 for a in range(self.config.action_num)]
            self.q_int[state] = [0.0 for a in range(self.config.action_num)]
            self.action_count[state] = [0 for a in range(self.config.action_num)]
            self.lifelong[state] = 1

    def iteration_q(self, mode: str):
        if mode == "ext":
            discount = self.config.q_ext_discount
            q_tbl = self.q_ext
            prob = self.config.q_ext_target_policy_prob
        elif mode == "int":
            discount = self.config.q_int_discount
            q_tbl = self.q_int
            prob = self.config.q_int_target_policy_prob
        else:
            raise UndefinedError(mode)
        delta = 0
        for _ in range(self.config.q_max_iteration):  # for safety
            delta = 0
            for state in self.trans.keys():
                for act in range(self.config.action_num):
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
                        reward = self.reward_ext[state][act][next_state]
                        done = self.done[state][act][next_state]
                        n_q = self.calc_next_q(q_tbl[next_state], prob, self.invalid_actions[next_state])
                        gain = reward + (1 - done) * discount * n_q
                        q += trans_prob * gain

                    delta = max(delta, abs(q_tbl[state][act] - q))
                    q_tbl[state][act] = q

            if delta < self.config.iteration_threshold:
                break
        else:
            logger.warning(f"iteration {self.config.q_max_iteration} over: {delta}, {mode}")

    def calc_next_q(self, q_tbl, prob: float, invalid_actions):
        q_max = max(q_tbl)

        if self.config.action_num == len(invalid_actions):
            # 有効アクションがない場合
            return 0

        if prob == 1:
            return q_max

        q_max_idx = [a for a, q in enumerate(q_tbl) if q == q_max and (a not in invalid_actions)]
        valid_actions = self.config.action_num - len(invalid_actions)
        if valid_actions == len(q_max_idx):
            prob = 1.0

        n_q = 0
        for a in range(self.config.action_num):
            if a in invalid_actions:
                continue
            elif a in q_max_idx:
                p = prob / len(q_max_idx)
            else:
                p = (1 - prob) / (valid_actions - len(q_max_idx))
            n_q += p * q_tbl[a]
        return n_q

    def sample_next_state(self, state, action):
        if state not in self.trans:
            return None
        n_s_list = list(self.trans[state][action].keys())
        if len(n_s_list) == 0:
            return None
        weights = list(self.trans[state][action].values())
        r_idx = random_choice_by_probs(weights)
        return n_s_list[r_idx]


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        batchs = self.memory.sample(self.batch_size, self.train_count)

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

            # --- int (sarsa)
            n_q = self.parameter.calc_next_q(
                self.parameter.q_int[n_state], self.config.q_int_target_policy_prob, next_invalid_actions
            )
            target_q = reward_int + (1 - done) * self.config.q_int_discount * n_q
            td_error_int = target_q - self.parameter.q_int[state][action]
            self.parameter.q_int[state][action] += self.config.q_int_lr * td_error_int

            # --- lifelong
            self.parameter.lifelong[state] *= self.config.lifelong_decrement_rate

            # --- action count update
            if self.distributed:
                self.parameter.action_count[state][action] += 1

            if self.train_count % self.config.iteration_interval == 0:
                self.parameter.iteration_q("ext")
                self.parameter.iteration_q("int")

            self.train_count += 1

        self.train_info = {
            "size": len(self.parameter.q_ext),
            "td_error_ext": td_error_ext,
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

    def on_start(self, worker: WorkerRun) -> None:
        self.parameter.iteration_q("ext")
        self.parameter.iteration_q("int")

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        self.episodic = {}
        return {}

    def call_policy(self, _state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        self.invalid_actions = invalid_actions
        self.state = to_str_observation(_state)
        self.parameter.init_q(self.state)

        q_ext = np.asarray(self.parameter.q_ext[self.state])
        q_int = np.asarray(self.parameter.q_int[self.state])
        q = self.config.q_ext_beta * q_ext + self.config.q_int_beta * q_int

        if self.training or self.rendering:
            # UCBのスケールに合わせてざっくり0～1になるようにする
            q = (q + 2) / 4

            # actionはUCBで決める
            N = sum(self.parameter.action_count[self.state])
            self.ucb_list = []
            for a in range(self.config.action_num):
                if a in invalid_actions:
                    self.ucb_list.append(-np.inf)
                    continue
                n = self.parameter.action_count[self.state][a]
                if n == 0:
                    self.ucb_list.append(np.inf)
                else:
                    ucb = q[a] + self.config.action_ucb_beta * np.sqrt(np.log(N) / n)
                    self.ucb_list.append(ucb)

        if self.training:
            self.action = get_random_max_index(self.ucb_list, invalid_actions)
            self.parameter.action_count[self.state][self.action] += 1
        else:
            self.action = get_random_max_index(q, invalid_actions)

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
        self.parameter.init_q(next_state)

        # 内部報酬
        episodic_reward = self._calc_episodic_reward(next_state)
        lifelong_reward = self._calc_lifelong_reward(next_state)
        reward_int = (episodic_reward * lifelong_reward) * 2 - 1

        if self.config.use_symlog:
            reward_ext = symlog(reward_ext)

        batch = {
            "state": self.state,
            "next_state": next_state,
            "action": self.action,
            "reward_ext": reward_ext,
            "reward_int": reward_int,
            "done": self.worker.done_reason == DoneTypes.TERMINATED,
            "invalid_actions": self.invalid_actions,
            "next_invalid_actions": next_invalid_actions,
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

    def render_terminal(self, worker: WorkerRun, **kwargs) -> None:
        self.parameter.init_state(self.state, self.action, None, self.invalid_actions, [])
        episodic_reward = self._calc_episodic_reward(self.state, update=False)
        lifelong_reward = self._calc_lifelong_reward(self.state)
        int_reward = episodic_reward * lifelong_reward
        print(f"int_reward {int_reward:.4f} = episodic {episodic_reward:.3f} * lifelong {lifelong_reward:.3f}")

        q_ext = self.parameter.q_ext[self.state]
        q_int = self.parameter.q_int[self.state]
        maxa = np.argmax(q_ext)

        def _render_sub(a: int) -> str:
            s = f"{q_ext[a]:8.5f}(ext), {q_int[a]:8.5f}(int)"
            s += f", {self.parameter.action_count[self.state][a]}n"
            s += f", ucb {self.ucb_list[a]:.3f}"
            n_state = self.parameter.sample_next_state(self.state, a)
            if n_state in self.parameter.reward_ext[self.state][a]:
                s += f", next {n_state}"
                s += f", reward_ext {self.parameter.reward_ext[self.state][a][n_state]:8.5f}"
                s += f", reward_int {self.parameter.reward_int[self.state][a][n_state]:8.5f}"
                s += f", done {100*self.parameter.done[self.state][a][n_state]:.1f}%"
            return s

        render_discrete_action(maxa, worker.env, self.config, _render_sub)
