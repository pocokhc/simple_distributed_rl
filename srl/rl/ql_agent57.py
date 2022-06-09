import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, cast

import numpy as np
from srl.base.define import RLObservationType
from srl.base.env.base import EnvRun
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import PriorityExperienceReplay
from srl.rl.functions.common import (
    calc_epsilon_greedy_probs,
    create_beta_list,
    create_epsilon_list,
    create_gamma_list,
    inverse_rescaling,
    random_choice_by_probs,
    render_discrete_action,
    rescaling,
    to_str_observation,
)

logger = logging.getLogger(__name__)


"""
DQN
    window_length          : x
    Fixed Target Q-Network : o
    Error clipping      : -
    Experience Replay   : o
    Frame skip          : -
    Annealing e-greedy  : x
    Reward clip         : x
    Image preprocessor  : -
Rainbow
    Double Q-learning        : o (config selection)
    Priority Experience Reply: o (config selection)
    Dueling Network          : -
    Multi-Step learning      : (retrace)
    Noisy Network            : -
    Categorical DQN          : -
Recurrent Replay Distributed DQN(R2D2)
    LSTM                     : -
    Value function rescaling : o (config selection)
Never Give Up(NGU)
    Intrinsic Reward : o (config selection)
    UVFA             : -
    Retrace          : o (config selection)
Agent57
    Meta controller(sliding-window UCB) : o (config selection)
    Intrinsic Reward split              : o
Other
    invalid_actions : TODO
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    # ハイパーパラメータ
    test_epsilon: float = 0.0
    test_beta: float = 0.0

    ext_lr: float = 0.01
    warmup_size: int = 10
    enable_rescale: bool = True
    enable_q_int_norm: bool = False

    # Priority Experience Replay
    capacity: int = 100_000
    memory_name: str = "ProportionalMemory"
    memory_warmup_size: int = 100
    memory_alpha: float = 0.6
    memory_beta_initial: float = 1.0
    memory_beta_steps: int = 1_000_000

    # retrace
    multisteps: int = 1
    retrace_h: float = 1

    # target model
    target_model_update_interval: int = 100
    enable_double_dqn: bool = True

    # ucb(160,0.5 or 3600,0.01)
    enable_actor: bool = True
    actor_num: int = 32
    ucb_window_size: int = 160  # UCB上限
    ucb_epsilon: float = 0.5  # UCBを使う確率
    ucb_beta: float = 1  # UCBのβ
    # actorを使わない場合の設定
    epsilon: float = 0.1
    gamma: float = 0.9
    beta: float = 0.1

    # intrinsic_reward
    enable_intrinsic_reward: bool = True  # 内部報酬を使うか
    int_lr: float = 0.01  # 学習率

    # episodic
    episodic_memory_capacity: int = 30000

    # lifelong
    lifelong_decrement_rate: float = 0.999  # 減少割合
    lifelong_reward_L: float = 5.0

    def __post_init__(self):
        super().__init__()

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.DISCRETE

    @staticmethod
    def getName() -> str:
        return "QL_Agent57"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.actor_num > 0
        assert self.multisteps >= 1
        assert self.memory_warmup_size < self.capacity


register(
    Config,
    __name__ + ":RemoteMemory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


# ------------------------------------------------------
# RemoteMemory
# ------------------------------------------------------
class RemoteMemory(PriorityExperienceReplay):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.init(
            self.config.memory_name,
            self.config.capacity,
            self.config.memory_alpha,
            self.config.memory_beta_initial,
            self.config.memory_beta_steps,
        )

    def sample(self, step: int) -> Tuple[int, Any, float]:
        indices, batchs, weights = self.memory.sample(1, step)
        return indices[0], batchs[0], weights[0]

    def update(self, index: int, batch: Any, priority: float) -> None:
        self.memory.update([index], [batch], [priority])


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.Q_ext = {}
        self.Q_ext_target = {}
        self.Q_int = {}
        self.Q_int_target = {}
        self.lifelong_C = {}
        self.Q_C = {}

        self.init_state("", [])

    def restore(self, data: Any) -> None:
        d = json.loads(data)
        self.Q_ext = d[0]
        self.Q_ext_target = d[0]
        self.Q_int = d[1]
        self.Q_int_target = d[1]
        self.lifelong_C = d[2]
        self.Q_C = d[3]

    def backup(self):
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
            self.Q_ext[state] = [-np.inf if a in invalid_actions else 0.0 for a in range(self.config.nb_actions)]
            self.Q_ext_target[state] = [
                -np.inf if a in invalid_actions else 0.0 for a in range(self.config.nb_actions)
            ]
            L = self.config.lifelong_reward_L
            if self.config.enable_rescale:
                L = rescaling(L)
            self.Q_int[state] = [0.0 if a in invalid_actions else L for a in range(self.config.nb_actions)]
            self.Q_int_target[state] = [0.0 if a in invalid_actions else L for a in range(self.config.nb_actions)]
            self.Q_C[state] = 0

    def calc_td_error(self, batch, Q, Q_target, rewards, enable_norm=False):

        dones = [False for _ in range(len(rewards))]
        dones[-1] = batch["done"]
        gamma = batch["gamma"]

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

            q = Q[state]
            n_q = Q[n_state]
            n_q_target = Q_target[n_state]

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
                # DoubleDQN: indexはonlineQから選び、値はtargetQを選ぶ
                if self.config.enable_double_dqn:
                    n_act_idx = np.argmax(n_q)
                else:
                    n_act_idx = np.argmax(n_q_target)
                P = n_q_target[n_act_idx]

                # --- original code
                if enable_norm and self.Q_C[n_state] > 0:
                    P = P / self.Q_C[n_state]
                # ---

                if self.config.enable_rescale:
                    P = inverse_rescaling(P)
                target_q = reward + gamma * P
            if self.config.enable_rescale:
                target_q = rescaling(target_q)

            td_error = target_q - Q[state][action]
            TQ += (gamma**n) * _retrace * td_error

        return TQ


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
        self.sync_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):

        if self.remote_memory.length() < self.config.memory_warmup_size:
            return {}

        index, batch, weight = self.remote_memory.sample(self.train_count)
        state = batch["states"][0]
        action = batch["actions"][0]
        ext_td_error = self.parameter.calc_td_error(
            batch,
            self.parameter.Q_ext,
            self.parameter.Q_ext_target,
            batch["ext_rewards"],
        )
        self.parameter.Q_ext[state][action] += self.config.ext_lr * ext_td_error * weight

        if self.config.enable_intrinsic_reward:
            int_td_error = self.parameter.calc_td_error(
                batch,
                self.parameter.Q_int,
                self.parameter.Q_int_target,
                batch["int_rewards"],
                enable_norm=True,
            )
            self.parameter.Q_int[state][action] += self.config.int_lr * int_td_error * weight
            self.parameter.Q_C[state] += 1
        else:
            int_td_error = 0

        # 外部Qを優先
        priority = abs(ext_td_error) + 0.0001
        # priority = abs(ext_td_error) + batch["beta"] * abs(int_td_error) + 0.0001
        self.remote_memory.update(index, batch, priority)

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            for k, v in self.parameter.Q_ext.items():
                self.parameter.Q_ext_target[k] = v
            for k, v in self.parameter.Q_int.items():
                self.parameter.Q_int_target[k] = v
            self.sync_count += 1

        self.train_count += 1

        return {
            "Q": len(self.parameter.Q_ext),
            "ext_td_error": ext_td_error,
            "int_td_error": int_td_error,
            "sync": self.sync_count,
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

        self.beta_list = create_beta_list(self.config.actor_num)
        self.gamma_list = create_gamma_list(self.config.actor_num)
        self.epsilon_list = create_epsilon_list(self.config.actor_num)

        # ucb
        self.actor_index = -1
        self.ucb_recent = []
        self.ucb_actors_count = [1 for _ in range(self.config.actor_num)]  # 1回は保証
        self.ucb_actors_reward = [0.0 for _ in range(self.config.actor_num)]

    def call_on_reset(self, state_: np.ndarray, invalid_actions: List[int]) -> None:
        state = to_str_observation(state_)

        if self.training:
            if self.config.enable_actor:
                # エピソード毎に actor を決める
                self.actor_index = self._calc_actor_index()
                self.epsilon = self.epsilon_list[self.actor_index]
                self.gamma = self.gamma_list[self.actor_index]
                self.beta = self.beta_list[self.actor_index]
            else:
                self.epsilon = self.config.epsilon
                self.gamma = self.config.gamma
                self.beta = self.config.beta
        else:
            self.epsilon = self.config.test_epsilon
            self.beta = self.config.test_beta

        if self.config.enable_intrinsic_reward:
            self.beta = 0

        self.recent_ext_rewards = [0.0 for _ in range(self.config.multisteps)]
        self.recent_int_rewards = [0.0 for _ in range(self.config.multisteps)]
        self.recent_actions = [random.randint(0, self.config.nb_actions - 1) for _ in range(self.config.multisteps)]
        self.recent_probs = [1.0 for _ in range(self.config.multisteps)]
        self.recent_states = ["" for _ in range(self.config.multisteps + 1)]
        self.recent_invalid_actions = [[] for _ in range(self.config.multisteps + 1)]

        self.recent_states.pop(0)
        self.recent_states.append(state)
        self.recent_invalid_actions.pop(0)
        self.recent_invalid_actions.append(invalid_actions)

        # sliding-window UCB 用に報酬を保存
        self.prev_episode_reward = 0.0

        # エピソード内での訪問回数
        self.episodic_C = {}

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
        return random.choice(np.where(ucbs == np.max(ucbs))[0])

    def call_policy(self, _state: np.ndarray, invalid_actions: List[int]) -> int:
        state = self.recent_states[-1]

        self.parameter.init_state(state, invalid_actions)
        q_ext = np.asarray(self.parameter.Q_ext[state])
        q_int = np.asarray(self.parameter.Q_int[state])
        q = q_ext + self.beta * q_int

        probs = calc_epsilon_greedy_probs(q, invalid_actions, self.epsilon, self.config.nb_actions)
        self.action = random_choice_by_probs(probs)
        self.prob = probs[self.action]
        return self.action

    def call_on_step(
        self,
        next_state: Any,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ) -> Dict:

        n_state = to_str_observation(next_state)
        self.recent_states.pop(0)
        self.recent_states.append(n_state)
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
                self.recent_states.pop(0)
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
        if self.recent_states[0] == "" and self.recent_states[1] == "":
            return

        batch = {
            "states": self.recent_states[:],
            "actions": self.recent_actions[:],
            "probs": self.recent_probs[:],
            "ext_rewards": self.recent_ext_rewards[:],
            "int_rewards": self.recent_int_rewards[:],
            "invalid_actions": self.recent_invalid_actions[:],
            "done": done,
            "gamma": self.gamma,
        }

        # priority
        if self.config.memory_name == "ReplayMemory":
            priority = 0
        elif not self.distributed:
            priority = 0
        else:
            td_error = self.parameter.calc_td_error(
                batch,
                self.parameter.Q_ext,
                self.parameter.Q_ext_target,
                self.recent_ext_rewards,
            )
            priority = abs(td_error) + 0.0001

        self.remote_memory.add(batch, priority)

        return priority

    def _calc_episodic_reward(self, state):
        if state not in self.episodic_C:
            self.episodic_C[state] = 0

        # 0回だと無限大になるので1回は保証する
        reward = 1 / np.sqrt(self.episodic_C[state] + 1)

        # 数える
        self.episodic_C[state] += 1

        return reward

    def _calc_lifelong_reward(self, state):
        # RNDと同じ発想で、回数を重ねると0に近づくようにする
        if state not in self.parameter.lifelong_C:
            self.parameter.lifelong_C[state] = self.config.lifelong_reward_L - 1.0  # 初期値
        reward = self.parameter.lifelong_C[state]

        # 0に近づける
        self.parameter.lifelong_C[state] *= self.config.lifelong_decrement_rate
        return reward + 1.0

    def call_render(self, env: EnvRun) -> None:
        state = self.recent_states[-1]
        invalid_actions = self.recent_invalid_actions[-1]
        self.parameter.init_state(state, invalid_actions)

        episodic_reward = self._calc_episodic_reward(state)
        lifelong_reward = self._calc_lifelong_reward(state)
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

        render_discrete_action(invalid_actions, maxa, env, _render_sub)
