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
from srl.rl.functions.common import (
    calc_epsilon_greedy_probs,
    create_beta_list,
    create_discount_list,
    create_epsilon_list,
    inverse_rescaling,
    random_choice_by_probs,
    render_discrete_action,
    rescaling,
    to_str_observation,
)
from srl.rl.memories.priority_experience_replay import PriorityExperienceReplay, PriorityExperienceReplayConfig
from srl.rl.schedulers.scheduler import SchedulerConfig

logger = logging.getLogger(__name__)


"""
DQN
    window_length          : -
    Fixed Target Q-Network : x
    Error clipping      : -
    Experience Replay   : o
    Frame skip          : -
    Annealing e-greedy  : x
    Reward clip         : x
    Image preprocessor  : -
Rainbow
    Double Q-learning        : TODO
    Priority Experience Reply: o
    Dueling Network(Advantage updating) : TODO
    Multi-Step learning      : o
    Noisy Network            : -
    Categorical DQN          : -
Recurrent Replay Distributed DQN(R2D2)
    LSTM                     : -
    Value function rescaling : o
Never Give Up(NGU)
    Intrinsic Reward : o
    UVFA             : -
    Retrace          : o
Agent57
    Meta controller(sliding-window UCB) : o
    Intrinsic Reward split              : o
Other
    invalid_actions : o
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig, PriorityExperienceReplayConfig):
    #: ε-greedy parameter for Test
    test_epsilon: float = 0.0
    test_beta: float = 0.0

    lr_ext: float = 0.01  # type: ignore , type OK
    enable_rescale: bool = False

    # retrace
    multisteps: int = 1
    retrace_h: float = 1

    # ucb(160,0.5 or 3600,0.01)
    enable_actor: bool = True
    actor_num: int = 32
    ucb_window_size: int = 160  # UCB上限
    ucb_epsilon: float = 0.5  # UCBを使う確率
    ucb_beta: float = 1  # UCBのβ
    # actorを使わない場合の設定
    epsilon: float = 0.1  # type: ignore , type OK
    discount: float = 0.9
    beta: float = 0.1

    # intrinsic_reward
    enable_intrinsic_reward: bool = True  # 内部報酬を使うか
    lr_int: float = 0.01  # type: ignore , type OK

    # episodic
    episodic_memory_capacity: int = 30000

    # lifelong
    lifelong_decrement_rate: float = 0.999  # 減少割合
    lifelong_reward_L: float = 5.0

    #: How to initialize Q table
    #:
    #: Parameters:
    #:   "": 0
    #:   "random": np.random.normal()
    q_init: str = ""

    def __post_init__(self):
        super().__post_init__()

        self.lr_ext: SchedulerConfig = SchedulerConfig(cast(float, self.lr_ext))
        self.lr_int: SchedulerConfig = SchedulerConfig(cast(float, self.lr_int))
        self.epsilon: SchedulerConfig = SchedulerConfig(cast(float, self.epsilon))
        self.memory.warmup_size = 10
        self.batch_size = 4

    @property
    def base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    @property
    def base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    def get_use_framework(self) -> str:
        return ""

    def getName(self) -> str:
        return "QL_Agent57"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_memory()
        assert self.actor_num > 0
        assert self.multisteps >= 1

    @property
    def info_types(self) -> dict:
        return {
            "size": {"type": int, "data": "last"},
            "ext_td_error": {},
            "int_td_error": {},
            "episodic_reward": {},
            "lifelong_reward": {},
            "int_reward": {},
            "priority": {},
            "lr_ext": {"data": "last"},
            "lr_int": {"data": "last"},
            "epsilon": {"data": "last"},
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
class Memory(PriorityExperienceReplay):
    pass


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
        self.Q_C = {}

        self.init_state("", [])

    def call_restore(self, data: Any, **kwargs) -> None:
        d = json.loads(data)
        self.Q_ext = d[0]
        self.Q_int = d[1]
        self.lifelong_C = d[2]
        self.Q_C = d[3]

    def call_backup(self, **kwargs):
        return json.dumps(
            [
                self.Q_ext,
                self.Q_int,
                self.lifelong_C,
                self.Q_C,
            ]
        )

    # ----------------------------------------

    def init_state(self, state, invalid_actions):
        if state not in self.Q_ext:
            if self.config.q_init == "random":
                self.Q_ext[state] = [
                    -np.inf if a in invalid_actions else np.random.normal() for a in range(self.config.action_num)
                ]
            else:
                self.Q_ext[state] = [-np.inf if a in invalid_actions else 0.0 for a in range(self.config.action_num)]
            L = self.config.lifelong_reward_L
            if self.config.enable_rescale:
                L = rescaling(L)
            self.Q_int[state] = [0.0 if a in invalid_actions else L for a in range(self.config.action_num)]
            self.Q_C[state] = 0

    def calc_td_error(self, batch, q_tbl, rewards, enable_norm=False):
        dones = [False for _ in range(len(rewards))]
        dones[-1] = batch["done"]
        discount = batch["discount"]

        TQ = 0
        _retrace = 1
        for n in range(len(rewards)):
            state = batch["states"][n]
            n_state = batch["states"][n + 1]
            invalid_actions = batch["invalid_actions"][n]
            n_invalid_actions = batch["invalid_actions"][n + 1]
            action = batch["actions"][n]
            mu_prob = batch["probs"][n]
            reward = rewards[n]
            done = dones[n]

            self.init_state(state, invalid_actions)
            self.init_state(n_state, n_invalid_actions)

            q = q_tbl[state]
            n_q = q_tbl[n_state]

            # retrace
            if n >= 1:
                if np.argmax(q) == action:
                    pi_prob = 1
                else:
                    pi_prob = 0
                _retrace *= self.config.retrace_h * np.minimum(1, pi_prob / mu_prob)
                if _retrace == 0:
                    break  # 0以降は伝搬しないので切りあげる

            if done:
                target_q = reward
            else:
                maxq = np.max(n_q)

                # --- original code
                if enable_norm and self.Q_C[n_state] > 0:
                    maxq = maxq / self.Q_C[n_state]
                # ---

                if self.config.enable_rescale:
                    maxq = inverse_rescaling(maxq)
                target_q = reward + discount * maxq
            if self.config.enable_rescale:
                target_q = rescaling(target_q)

            td_error = target_q - q_tbl[state][action]
            TQ += (discount**n) * _retrace * td_error

        return TQ


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_ext_sch = self.config.lr_ext.create_schedulers()
        self.lr_int_sch = self.config.lr_int.create_schedulers()

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        indices, batchs, weights = self.memory.sample(self.batch_size, self.train_count)

        lr_ext = self.lr_ext_sch.get_and_update_rate(self.train_count)
        lr_int = self.lr_int_sch.get_and_update_rate(self.train_count)
        ext_td_errors = []
        int_td_errors = []
        for i in range(self.config.batch_size):
            state = batchs[i]["states"][0]
            action = batchs[i]["actions"][0]
            ext_td_error = self.parameter.calc_td_error(
                batchs[i],
                self.parameter.Q_ext,
                batchs[i]["ext_rewards"],
            )
            self.parameter.Q_ext[state][action] += lr_ext * ext_td_error * weights[i]
            ext_td_errors.append(ext_td_error)

            if self.config.enable_intrinsic_reward:
                int_td_error = self.parameter.calc_td_error(
                    batchs[i],
                    self.parameter.Q_int,
                    batchs[i]["int_rewards"],
                    enable_norm=True,
                )
                self.parameter.Q_int[state][action] += lr_int * int_td_error * weights[i]
                self.parameter.Q_C[state] += 1
            else:
                int_td_error = 0
            int_td_errors.append(int_td_error)

            self.train_count += 1

        # 外部Qを優先
        # priority = abs(ext_td_error) + batch["beta"] * abs(int_td_error)
        self.memory.update((indices, batchs, np.abs(ext_td_errors)))

        self.train_info = {
            "size": len(self.parameter.Q_ext),
            "ext_td_error": np.mean(ext_td_errors),
            "int_td_error": np.mean(int_td_errors),
            "lr_ext": lr_ext,
            "lr_int": lr_int,
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.beta_list = create_beta_list(self.config.actor_num)
        self.discount_list = create_discount_list(self.config.actor_num)
        self.epsilon_list = create_epsilon_list(self.config.actor_num)

        self.epsilon_scheduler = self.config.epsilon.create_schedulers()

        # ucb
        self.actor_index = -1
        self.ucb_recent = []
        self.ucb_actors_count = [1 for _ in range(self.config.actor_num)]  # 1回は保証
        self.ucb_actors_reward = [0.0 for _ in range(self.config.actor_num)]

    def call_on_reset(self, state_: np.ndarray, invalid_actions: List[int]) -> dict:
        state = to_str_observation(state_)

        if self.training:
            if self.config.enable_actor:
                # エピソード毎に actor を決める
                self.actor_index = self._calc_actor_index()
                self.epsilon = self.epsilon_list[self.actor_index]
                self.discount = self.discount_list[self.actor_index]
                self.beta = self.beta_list[self.actor_index]
            else:
                self.epsilon = self.epsilon_scheduler.get_and_update_rate(self.total_step)
                self.discount = self.config.discount
                self.beta = self.config.beta
        else:
            self.epsilon = self.config.test_epsilon
            self.beta = self.config.test_beta

        if self.config.enable_intrinsic_reward:
            self.beta = 0

        self._recent_states = ["" for _ in range(self.config.multisteps + 1)]
        self.recent_ext_rewards = [0.0 for _ in range(self.config.multisteps)]
        self.recent_int_rewards = [0.0 for _ in range(self.config.multisteps)]
        self.recent_actions = [random.randint(0, self.config.action_num - 1) for _ in range(self.config.multisteps)]
        self.recent_probs = [1.0 for _ in range(self.config.multisteps)]
        self.recent_invalid_actions = [[] for _ in range(self.config.multisteps + 1)]

        self._recent_states.pop(0)
        self._recent_states.append(state)
        self.recent_invalid_actions.pop(0)
        self.recent_invalid_actions.append(invalid_actions)

        # sliding-window UCB 用に報酬を保存
        self.prev_episode_reward = 0.0

        # エピソード内での訪問回数
        self.episodic_C = {}

        return {}

    # (sliding-window UCB)
    def _calc_actor_index(self) -> int:
        # UCB計算用に保存
        if self.actor_index != -1:
            self.ucb_recent.append(
                (
                    self.actor_index,
                    self.prev_episode_reward,
                )
            )
            self.ucb_actors_count[self.actor_index] += 1
            self.ucb_actors_reward[self.actor_index] += self.prev_episode_reward
            if len(self.ucb_recent) >= self.config.ucb_window_size:
                d = self.ucb_recent.pop(0)
                self.ucb_actors_count[d[0]] -= 1
                self.ucb_actors_reward[d[0]] -= d[1]

        N = len(self.ucb_recent)

        # 全て１回は実行
        if N < self.config.actor_num:
            return N

        # ランダムでactorを決定
        if random.random() < self.config.ucb_epsilon:
            return random.randint(0, self.config.actor_num - 1)

        # UCB値を計算
        ucbs = []
        for i in range(self.config.actor_num):
            n = self.ucb_actors_count[i]
            u = self.ucb_actors_reward[i] / n
            ucb = u + self.config.ucb_beta * np.sqrt(np.log(N) / n)
            ucbs.append(ucb)

        # UCB値最大のポリシー（複数あればランダム）
        return np.random.choice(np.where(ucbs == np.max(ucbs))[0])

    def call_policy(self, _state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        state = self._recent_states[-1]

        self.parameter.init_state(state, invalid_actions)
        q_ext = np.asarray(self.parameter.Q_ext[state])
        q_int = np.asarray(self.parameter.Q_int[state])
        q = q_ext + self.beta * q_int

        probs = calc_epsilon_greedy_probs(q, invalid_actions, self.epsilon, self.config.action_num)
        self.action = random_choice_by_probs(probs)
        self.prob = probs[self.action]
        return self.action, {}

    def call_on_step(
        self,
        next_state: Any,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ) -> dict:
        self.prev_episode_reward += reward
        n_state = to_str_observation(next_state)
        self._recent_states.pop(0)
        self._recent_states.append(n_state)
        self.recent_invalid_actions.pop(0)
        self.recent_invalid_actions.append(next_invalid_actions)

        if not self.training:
            return {}

        # 内部報酬
        if self.config.enable_intrinsic_reward:
            episodic_reward = self._calc_episodic_reward(n_state)
            lifelong_reward = self._calc_lifelong_reward(n_state)
            int_reward = episodic_reward * lifelong_reward
        else:
            episodic_reward = 0
            lifelong_reward = 0
            int_reward = 0

        self.recent_actions.pop(0)
        self.recent_actions.append(self.action)
        self.recent_probs.pop(0)
        self.recent_probs.append(self.prob)
        self.recent_ext_rewards.pop(0)
        self.recent_ext_rewards.append(reward)
        self.recent_int_rewards.pop(0)
        self.recent_int_rewards.append(int_reward)

        priority = self._add_memory(done)

        if done:
            # 残りstepも追加
            for _ in range(len(self.recent_ext_rewards) - 1):
                self._recent_states.pop(0)
                self.recent_invalid_actions.pop(0)
                self.recent_actions.pop(0)
                self.recent_probs.pop(0)
                self.recent_ext_rewards.pop(0)
                self.recent_int_rewards.pop(0)

                self._add_memory(done)

        return {
            "episodic_reward": episodic_reward,
            "lifelong_reward": lifelong_reward,
            "int_reward": int_reward,
            "priority": priority,
        }

    def _add_memory(self, done):
        if self._recent_states[0] == "" and self._recent_states[1] == "":
            return

        batch = {
            "states": self._recent_states[:],
            "actions": self.recent_actions[:],
            "probs": self.recent_probs[:],
            "ext_rewards": self.recent_ext_rewards[:],
            "int_rewards": self.recent_int_rewards[:],
            "invalid_actions": self.recent_invalid_actions[:],
            "done": done,
            "discount": self.discount,
        }

        # priority
        if not self.distributed:
            priority = 0
        elif not self.config.memory.requires_priority():
            priority = 0
        else:
            td_error = self.parameter.calc_td_error(
                batch,
                self.parameter.Q_ext,
                self.recent_ext_rewards,
            )
            priority = abs(td_error)

        self.memory.add(batch, priority)
        return priority

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

    def render_terminal(self, worker, **kwargs) -> None:
        state = self._recent_states[-1]
        invalid_actions = self.recent_invalid_actions[-1]
        self.parameter.init_state(state, invalid_actions)

        episodic_reward = self._calc_episodic_reward(state, update=False)
        lifelong_reward = self._calc_lifelong_reward(state, update=False)
        int_reward = episodic_reward * lifelong_reward
        print(f"int_reward {int_reward:.4f} = episodic {episodic_reward:.3f} * lifelong {lifelong_reward:.3f}")
        q_ext = np.asarray(self.parameter.Q_ext[state])
        q_int = np.asarray(self.parameter.Q_int[state])
        q = q_ext + self.beta * q_int
        if self.config.enable_rescale:
            q_ext = inverse_rescaling(q_ext)
            q_int = inverse_rescaling(q_int)
            q = inverse_rescaling(q)
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:8.5f} = {q_ext[a]:8.5f} + {self.beta:.3f} * {q_int[a]:8.5f}"

        render_discrete_action(maxa, worker.env, self.config, _render_sub)
