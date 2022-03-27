import json
import logging
import random
from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import numpy as np
from srl.base.rl import RLParameter, RLRemoteMemory, RLTrainer, RLWorker, TableConfig
from srl.rl.functions.common import (
    calc_epsilon_greedy_probs,
    create_beta_list,
    create_epsilon_list,
    create_gamma_list,
    inverse_rescaling,
    rescaling,
)
from srl.rl.memory import factory
from srl.rl.registory import register

logger = logging.getLogger(__name__)


"""
DQN
    window_length       : x
    Target Network      : o
    Huber loss function : -
    Delay update Target Network: o
    Experience Replay   : o
    Frame skip          : x
    Annealing e-greedy  : x (sliding-window UCB)
    Reward clip         : x
    Image preprocessor  : x
Rainbow
    Double DQN               : o
    Priority Experience Reply: o
    Dueling Network          : -
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
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(TableConfig):

    # ハイパーパラメータ
    test_epsilon: float = 0.0001
    ext_lr: float = 0.1  # 学習率

    # Priority Experience Replay
    capacity: int = 100_000
    memory_name: str = "RankBaseMemory"
    memory_warmup_size: int = 1000
    memory_alpha: float = 0.6
    memory_beta_initial: float = 0.4
    memory_beta_steps: int = 1_000_000

    warmup_size: int = 10
    actor_num: int = 32  # ポリシー数(0で無効)
    use_rescale: bool = True  # rescaleするか
    enable_q_int_norm: bool = False

    # retrace
    multisteps: int = 1
    retrace_h: float = 1

    # target
    enable_target: bool = False
    enable_double_dqn: bool = False
    target_model_update_interval: int = 100

    # ucb(160,0.5 or 3600,0.01)
    ucb_window_size: int = 160  # UCB上限
    ucb_epsilon: float = 0.5  # UCBを使う確率
    ucb_beta: float = 1  # UCBのβ

    # intrinsic_reward
    use_intrinsic_reward: bool = True  # 内部報酬を使うか
    int_lr: float = 0.1  # 学習率

    # episodic
    episodic_memory_capacity: int = 30000

    # lifelong
    lifelong_decrement_rate: float = 0.999  # 減少割合
    lifelong_reward_L: float = 5.0

    @staticmethod
    def getName() -> str:
        return "QL_agent57"


register(Config, __name__)


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

        self.init_state("", [a for a in range(self.config.nb_actions)])

        self.beta_list = create_beta_list(self.config.actor_num)
        self.gamma_list = create_gamma_list(self.config.actor_num)
        self.epsilon_list = create_epsilon_list(self.config.actor_num)

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

    def init_state(self, state, valid_actions):
        if state not in self.Q_ext:
            self.Q_ext[state] = [0 if a in valid_actions else -np.inf for a in range(self.config.nb_actions)]
            self.Q_ext_target[state] = [0 if a in valid_actions else -np.inf for a in range(self.config.nb_actions)]
            self.Q_int[state] = [self.config.lifelong_reward_L for a in range(self.config.nb_actions)]
            self.Q_int_target[state] = [self.config.lifelong_reward_L for a in range(self.config.nb_actions)]
            self.Q_C[state] = 0

    def get_q(self, state, valid_actions, beta):
        self.init_state(state, valid_actions)
        q_ext = np.asarray(self.Q_ext[state])
        q_int = np.asarray(self.Q_int[state])
        q = q_ext + beta * q_int
        return q

    def calc_td_error(self, batch, Q, Q_target, rewards, enable_norm=False):

        dones = [False for _ in range(len(rewards))]
        dones[-1] = batch["done"]
        beta = batch["beta"]
        gamma = batch["gamma"]

        TQ = 0
        _retrace = 1
        for n in range(len(rewards)):
            s = batch["states"][n]
            n_s = batch["states"][n + 1]
            action = batch["actions"][n]
            mu_prob = batch["probs"][n]
            reward = rewards[n]
            done = dones[n]
            valid_action = batch["valid_actions"][n]
            next_valid_action = batch["valid_actions"][n + 1]

            q = self.get_q(s, valid_action, beta)
            n_q = self.get_q(n_s, next_valid_action, beta)

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
                n_act_idx = np.argmax(n_q)
                P = Q_target[n_s][n_act_idx]

                # ---
                if enable_norm and self.Q_C[n_s] > 0:
                    P = P / self.Q_C[n_s]
                # ---

                P = inverse_rescaling(P)
                target_q = reward + gamma * P
            target_q = rescaling(target_q)

            td_error = target_q - Q[s][action]
            TQ += (gamma**n) * _retrace * td_error

        return TQ


# ------------------------------------------------------
# RemoteMemory
# ------------------------------------------------------
class RemoteMemory(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.memory = factory.create(
            self.config.memory_name,
            {
                "capacity": self.config.capacity,
                "alpha": self.config.memory_alpha,
                "beta_initial": self.config.memory_beta_initial,
                "beta_steps": self.config.memory_beta_steps,
            },
        )

    def length(self) -> int:
        return len(self.memory)

    def restore(self, data: Any) -> None:
        self.memory.restore(data)

    def backup(self):
        return self.memory.backup()

    # --------------------

    def add(self, batch, priority):
        self.memory.add(batch, priority)

    def sample(self, step: int) -> Tuple[int, Any, float]:
        indexes, batchs, weights = self.memory.sample(1, step)
        return indexes[0], batchs[0], weights[0]

    def update(self, index: int, batch: Any, priority: float) -> None:
        self.memory.update([index], [batch], [priority])


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.memory = cast(RemoteMemory, self.memory)

        self.train_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):

        if self.memory.length() < self.config.memory_warmup_size:
            return {}

        index, batch, weight = self.memory.sample(self.train_count)
        state = batch["states"][0]
        action = batch["actions"][0]
        ext_td_error = self.parameter.calc_td_error(
            batch,
            self.parameter.Q_ext,
            self.parameter.Q_ext_target,
            batch["ext_rewards"],
        )
        self.parameter.Q_ext[state][action] += self.config.ext_lr * ext_td_error * weight
        int_td_error = self.parameter.calc_td_error(
            batch,
            self.parameter.Q_int,
            self.parameter.Q_int_target,
            batch["int_rewards"],
            enable_norm=True,
        )
        self.parameter.Q_int[state][action] += self.config.int_lr * int_td_error * weight
        self.parameter.Q_C[state] += 1

        # 外部Qを優先
        priority = abs(ext_td_error) + 0.0001
        # priority = abs(ext_td_error) + batch["beta"] * abs(int_td_error) + 0.0001
        self.memory.update(index, batch, priority)

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            for k, v in self.parameter.Q_ext.items():
                self.parameter.Q_ext_target[k] = v
            for k, v in self.parameter.Q_int.items():
                self.parameter.Q_int_target[k] = v

        self.train_count += 1

        return {
            "Q": len(self.parameter.Q_ext),
            "ext_td_error": ext_td_error,
            "int_td_error": int_td_error,
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.memory = cast(RemoteMemory, self.memory)

        # ucb
        self.actor_index = -1
        self.ucb_recent = []
        self.ucb_actors_count = [1 for _ in range(self.config.actor_num)]  # 1回は保証
        self.ucb_actors_reward = [0.0 for _ in range(self.config.actor_num)]

    def on_reset(self, state: np.ndarray, valid_actions: List[int], _) -> None:
        if self.training:
            # エピソード毎に actor を決める
            self.actor_index = self._calc_actor_index()
            self.gamma = self.parameter.gamma_list[self.actor_index]
            self.beta = self.parameter.beta_list[self.actor_index]
            self.epsilon = self.parameter.epsilon_list[self.actor_index]
        else:
            self.beta = 0.0
            self.epsilon = self.config.test_epsilon

        self.recent_ext_rewards = [0.0 for _ in range(self.config.multisteps)]
        self.recent_int_rewards = [0.0 for _ in range(self.config.multisteps)]
        self.recent_actions = [random.randint(0, self.config.nb_actions - 1) for _ in range(self.config.multisteps)]
        self.recent_probs = [1.0 for _ in range(self.config.multisteps)]
        self.recent_states = ["" for _ in range(self.config.multisteps + 1)]
        self.recent_valid_actions = [[] for _ in range(self.config.multisteps + 1)]

        s = str(state.tolist())
        self.recent_states.pop(0)
        self.recent_states.append(s)
        self.recent_valid_actions.pop(0)
        self.recent_valid_actions.append(valid_actions)

        self.parameter.init_state(s, valid_actions)

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

    def policy(self, state: np.ndarray, valid_actions: List[int], _) -> Tuple[int, Any]:
        s = str(state.tolist())
        q = self.parameter.get_q(s, valid_actions, self.beta)
        probs = calc_epsilon_greedy_probs(q, valid_actions, self.epsilon, self.config.nb_actions)
        action = random.choices([a for a in range(self.config.nb_actions)], weights=probs)[0]

        return action, (action, probs[action])

    def on_step(
        self,
        state: np.ndarray,
        action_: Any,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        valid_actions: List[int],
        next_valid_actions: List[int],
        _,
    ):
        if not self.training:
            return {}

        action = action_[0]
        prob = action_[1]
        n_state = str(next_state.tolist())
        self.parameter.init_state(n_state, next_valid_actions)

        # 内部報酬
        episodic_reward = self._calc_episodic_reward(n_state)
        lifelong_reward = self._calc_lifelong_reward(n_state)
        int_reward = episodic_reward * lifelong_reward

        self.recent_states.pop(0)
        self.recent_states.append(n_state)
        self.recent_actions.pop(0)
        self.recent_actions.append(action)
        self.recent_probs.pop(0)
        self.recent_probs.append(prob)
        self.recent_ext_rewards.pop(0)
        self.recent_ext_rewards.append(reward)
        self.recent_int_rewards.pop(0)
        self.recent_int_rewards.append(int_reward)
        self.recent_valid_actions.pop(0)
        self.recent_valid_actions.append(next_valid_actions)

        priority = self._add_memory(done)

        if done:
            # 残りstepも追加
            for _ in range(len(self.recent_ext_rewards) - 1):
                self.recent_states.pop(0)
                self.recent_actions.pop(0)
                self.recent_probs.pop(0)
                self.recent_ext_rewards.pop(0)
                self.recent_int_rewards.pop(0)

                self._add_memory(done)

        return {
            "priority": priority,
            "int_reward": int_reward,
        }

    def _add_memory(self, done):
        if self.recent_states[0] == "" and self.recent_states[1] == "":
            return

        batch = {
            "states": self.recent_states[:],
            "valid_actions": self.recent_valid_actions[:],
            "actions": self.recent_actions[:],
            "probs": self.recent_probs[:],
            "ext_rewards": self.recent_ext_rewards[:],
            "int_rewards": self.recent_int_rewards[:],
            "done": done,
            "gamma": self.gamma,
            "beta": self.beta,
        }

        # priority
        if self.config.memory_name == "ReplayMemory":
            priority = 1
        else:
            td_error = self.parameter.calc_td_error(
                batch,
                self.parameter.Q_ext,
                self.parameter.Q_ext_target,
                self.recent_ext_rewards,
            )
            priority = abs(td_error) + 0.0001

        self.memory.add(batch, priority)

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

    def render(self, state: np.ndarray, valid_actions: List[int], action_to_str) -> None:
        s = str(state.tolist())
        episodic_reward = self._calc_episodic_reward(s)
        lifelong_reward = self._calc_lifelong_reward(s)
        int_reward = episodic_reward * lifelong_reward
        print(f"int_reward {int_reward:7.5f}(episodic {episodic_reward:7.5f}, lifelong {lifelong_reward:7.5f})")
        q = self.parameter.get_q(s, valid_actions, self.beta)
        q = inverse_rescaling(q)
        maxa = np.argmax(q)
        for a in range(self.config.nb_actions):
            if a not in valid_actions:
                s = "x"
            else:
                s = " "
            if a == maxa:
                s += "*"
            else:
                s += " "
            s += f"{action_to_str(a)}: {q[a]:7.5f}"
            print(s)


if __name__ == "__main__":
    pass
