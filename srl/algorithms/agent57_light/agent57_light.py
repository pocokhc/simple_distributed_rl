import collections
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np

from srl.base.rl.algorithms.base_dqn import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import RLProcessor
from srl.rl import functions as funcs
from srl.rl.memories.priority_experience_replay import (
    PriorityExperienceReplay,
    RLConfigComponentPriorityExperienceReplay,
)
from srl.rl.models.config.dueling_network import DuelingNetworkConfig
from srl.rl.models.config.framework_config import RLConfigComponentFramework
from srl.rl.models.config.input_config import RLConfigComponentInput
from srl.rl.models.config.mlp_block import MLPBlockConfig
from srl.rl.schedulers.scheduler import SchedulerConfig

logger = logging.getLogger(__name__)

"""
DQN
    window_length          : -
    Fixed Target Q-Network : o
    Error clipping      : o
    Experience Replay   : o
    Frame skip          : -
    Annealing e-greedy  : x
    Reward clip         : x
    Image preprocessor  : -
Rainbow
    Double DQN               : o
    Priority Experience Reply: o
    Dueling Network          : o
    Multi-Step learning      : x
    Noisy Network            : x
    Categorical DQN          : x
Recurrent Replay Distributed DQN(R2D2)
    LSTM                     : x
    Value function rescaling : o
Never Give Up(NGU)
    Intrinsic Reward : o
    UVFA             : o
    Retrace          : x
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
class Config(
    RLConfig,
    RLConfigComponentPriorityExperienceReplay,
    RLConfigComponentFramework,
    RLConfigComponentInput,
):
    """
    <:ref:`RLConfigComponentPriorityExperienceReplay`>
    <:ref:`RLConfigComponentFramework`>
    <:ref:`RLConfigComponentInput`>
    """

    #: ε-greedy parameter for Test
    test_epsilon: float = 0
    #: intrinsic reward rate for Test
    test_beta: float = 0

    #: <:ref:`scheduler`> Learning rate
    lr_ext: Union[float, SchedulerConfig] = 0.0001
    #: <:ref:`scheduler`> Intrinsic network Learning rate
    lr_int: Union[float, SchedulerConfig] = 0.0001
    #: Synchronization interval to Target network
    target_model_update_interval: int = 1500

    #: enable DoubleDQN
    enable_double_dqn: bool = True
    #: enable rescaling
    enable_rescale: bool = False

    #: <:ref:`DuelingNetworkConfig`> hidden layer
    hidden_block: DuelingNetworkConfig = field(init=False, default_factory=lambda: DuelingNetworkConfig())

    #: ucb(160,0.5 or 3600,0.01)
    actor_num: int = 32
    #: UCB上限
    ucb_window_size: int = 3600
    #: UCBを使う確率
    ucb_epsilon: float = 0.01
    #: UCBのβ
    ucb_beta: float = 1

    #: enable intrinsic reward
    enable_intrinsic_reward: bool = True

    #: <:ref:`scheduler`> Episodic Learning rate
    episodic_lr: Union[float, SchedulerConfig] = 0.0005
    #: [episodic] k
    episodic_count_max: int = 10
    #: [episodic] epsilon
    episodic_epsilon: float = 0.001
    #: [episodic] cluster_distance
    episodic_cluster_distance: float = 0.008
    #: [episodic] capacity
    episodic_memory_capacity: int = 30000
    #: [episodic] 疑似カウント定数(c)
    episodic_pseudo_counts: float = 0.1
    #: <:ref:`MLPBlockConfig`> [episodic] emb block
    episodic_emb_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())
    #: <:ref:`MLPBlockConfig`> [episodic] out block
    episodic_out_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())

    #: <:ref:`scheduler`> Lifelong Learning rate
    lifelong_lr: Union[float, SchedulerConfig] = 0.0005
    #: [lifelong] L
    lifelong_max: float = 5.0
    #: <:ref:`MLPBlockConfig`> [lifelong] hidden block
    lifelong_hidden_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())

    #: [UVFA] input ext reward
    input_ext_reward: bool = True
    #: [UVFA] input int reward
    input_int_reward: bool = False
    #: [UVFA] input action
    input_action: bool = False

    #: Not use internal rewards to calculate priority
    disable_int_priority: bool = False
    #: dummy_state_val
    dummy_state_val: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.set_proportional_memory()
        self.hidden_block.set_dueling_network((512,))
        self.episodic_emb_block.set(
            (32,),
            activation="relu",
            # kernel_initializer="he_normal",
            # dense_kwargs={"bias_initializer": keras.initializers.constant(0.001)},
        )
        self.episodic_out_block.set((128,))
        self.lifelong_hidden_block.set((128,))

    def get_processors(self) -> List[RLProcessor]:
        return RLConfigComponentInput.get_processors(self)

    def get_framework(self) -> str:
        return RLConfigComponentFramework.get_framework(self)

    def get_name(self) -> str:
        return "Agent57_light"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_memory()
        self.assert_params_framework()
        self.assert_params_input()


# ------------------------------------------------------
# Memory
# ------------------------------------------------------
class Memory(PriorityExperienceReplay):
    pass


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class CommonInterfaceParameter(RLParameter[Config], ABC):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

    @abstractmethod
    def predict_q_ext_online(self, x) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def predict_q_ext_target(self, x) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def predict_q_int_online(self, x) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def predict_q_int_target(self, x) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def predict_emb(self, x) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def predict_lifelong_target(self, x) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def predict_lifelong_train(self, x) -> np.ndarray:
        raise NotImplementedError()

    def change_batchs_format(self, batchs):
        (
            states,
            n_states,
            onehot_actions,
            _,
            rewards_ext,
            rewards_int,
            dones,
            prev_onehot_actions,
            prev_rewards_ext,
            prev_rewards_int,
            actor_idx_list,
        ) = zip(*batchs)
        return (
            np.asarray(states),
            np.asarray(n_states),
            np.asarray(onehot_actions),
            [e for b in batchs for e in b[3]],
            [i for i, b in enumerate(batchs) for e in b[3]],
            np.array(rewards_ext, dtype=np.float32)[..., np.newaxis],
            np.array(rewards_int, dtype=np.float32)[..., np.newaxis],
            np.array(dones, dtype=np.float32),
            np.asarray(prev_onehot_actions),
            np.array(prev_rewards_ext, dtype=np.float32)[..., np.newaxis],
            np.array(prev_rewards_int, dtype=np.float32)[..., np.newaxis],
            np.array(actor_idx_list),
            np.identity(self.config.actor_num, dtype=np.float32)[[a for a in actor_idx_list]],
        )

    def calc_target_q(
        self,
        is_ext: bool,
        rewards,  # (batch)
        #
        n_states,
        rewards_ext,  # (batch, 1)
        rewards_int,  # (batch, 1)
        actions_onehot,
        actor_idx_onehot,
        next_invalid_actions_idx,
        next_invalid_actions,
        dones,  # (batch)
        batch_discount,  # (batch)
    ):
        batch_size = len(rewards)
        _inputs = [
            n_states,
            rewards_ext,
            rewards_int,
            actions_onehot,
            actor_idx_onehot,
        ]
        if is_ext:
            n_q_target = self.predict_q_ext_target(_inputs)
        else:
            n_q_target = self.predict_q_int_target(_inputs)

        # DoubleDQN: indexはonlineQから選び、値はtargetQを選ぶ
        if self.config.enable_double_dqn:
            if is_ext:
                n_q_online = self.predict_q_ext_online(_inputs)
            else:
                n_q_online = self.predict_q_int_online(_inputs)
            n_q_online[next_invalid_actions_idx, next_invalid_actions] = np.min(n_q_online)
            n_act_idx = np.argmax(n_q_online, axis=1)
            maxq = n_q_target[np.arange(batch_size), n_act_idx]
        else:
            n_q_target[next_invalid_actions_idx, next_invalid_actions] = np.min(n_q_target)
            maxq = np.max(n_q_target, axis=1)

        if self.config.enable_rescale:
            maxq = funcs.inverse_rescaling(maxq)

        # --- Q値を計算
        target_q = rewards + dones * batch_discount * maxq

        if self.config.enable_rescale:
            target_q = funcs.rescaling(target_q)

        return target_q


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker[Config, CommonInterfaceParameter]):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: CommonInterfaceParameter = self.parameter

        self.dummy_state = np.full(self.config.observation_space.shape, self.config.dummy_state_val, dtype=np.float32)
        self.discount = 0

        # actor
        self.beta_list = funcs.create_beta_list(self.config.actor_num)
        self.epsilon_list = funcs.create_epsilon_list(self.config.actor_num)
        self.discount_list = funcs.create_discount_list(self.config.actor_num)

        # ucb
        self.actor_index = -1
        self.ucb_recent = []
        self.ucb_actors_count = [1 for _ in range(self.config.actor_num)]  # 1回は保証
        self.ucb_actors_reward = [0.0 for _ in range(self.config.actor_num)]

    def on_reset(self, worker):
        if self.training:
            # エピソード毎に actor を決める
            self.actor_index = self._calc_actor_index()
            self.beta = self.beta_list[self.actor_index]
            self.epsilon = self.epsilon_list[self.actor_index]
            self.discount = self.discount_list[self.actor_index]
        else:
            self.actor_index = 0
            self.epsilon = self.config.test_epsilon
            self.beta = self.config.test_beta

        a0 = random.randint(0, self.config.action_space.n - 1)
        self.prev_onehot_action = np.identity(self.config.action_space.n, dtype=np.float32)[a0]
        self.prev_reward_ext = 0
        self.prev_reward_int = 0

        # Q値取得用
        self.onehot_actor_idx = np.identity(self.config.actor_num, dtype=np.float32)[self.actor_index][np.newaxis, ...]

        # sliding-window UCB 用に報酬を保存
        self.episode_reward = 0.0

        # エピソードメモリ(エピソード毎に初期化)
        self.episodic_memory = collections.deque(maxlen=self.config.episodic_memory_capacity)

        self.info["epsilon"] = self.epsilon
        self.info["beta"] = self.beta
        self.info["discount"] = self.discount

    # (sliding-window UCB)
    def _calc_actor_index(self) -> int:
        # UCB計算用に保存
        if self.actor_index != -1:
            self.ucb_recent.append(
                (
                    self.actor_index,
                    self.episode_reward,
                )
            )
            self.ucb_actors_count[self.actor_index] += 1
            self.ucb_actors_reward[self.actor_index] += self.episode_reward
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
        return funcs.get_random_max_index(ucbs)

    def policy(self, worker) -> int:

        in_ = [
            worker.state[np.newaxis, ...],
            np.array([[self.prev_reward_ext]], dtype=np.float32),
            np.array([[self.prev_reward_int]], dtype=np.float32),
            self.prev_onehot_action[np.newaxis, ...],
            self.onehot_actor_idx,
        ]
        self.q_ext = self.parameter.predict_q_ext_online(in_)[0]
        self.q_int = self.parameter.predict_q_int_online(in_)[0]
        self.q = self.q_ext + self.beta * self.q_int

        invalid_actions = worker.get_invalid_actions()
        if random.random() < self.epsilon:
            action = random.choice([a for a in range(self.config.action_space.n) if a not in invalid_actions])
        else:
            self.q[invalid_actions] = -np.inf
            action = int(np.argmax(self.q))

        self.onehot_action = np.identity(self.config.action_space.n, dtype=np.float32)[action]
        return action

    def on_step(self, worker):
        next_state = worker.state
        reward_ext = worker.reward
        next_invalid_actions = worker.get_invalid_actions()
        self.episode_reward += reward_ext

        # 内部報酬
        if self.config.enable_intrinsic_reward:
            n_s = next_state[np.newaxis, ...]
            episodic_reward = self._calc_episodic_reward(n_s)
            lifelong_reward = self._calc_lifelong_reward(n_s)
            reward_int = episodic_reward * lifelong_reward
            self.info["episodic"] = episodic_reward
            self.info["lifelong"] = lifelong_reward
            self.info["reward_int"] = reward_int
        else:
            reward_int = 0.0

        prev_onehot_action = self.prev_onehot_action
        prev_reward_ext = self.prev_reward_ext
        prev_reward_int = self.prev_reward_int
        self.prev_onehot_action = self.onehot_action
        self.prev_reward_ext = reward_ext
        self.prev_reward_int = reward_int

        if not self.training:
            return

        """
        [
            state,
            next_state,
            action,
            next_invalid_actions,
            reward_ext,
            reward_int,
            done,
            prev_action,
            prev_reward_ext,
            prev_reward_int,
            actor_idx_onehot,
        ]
        """
        batch = [
            worker.prev_state,
            next_state,
            self.onehot_action,
            next_invalid_actions,
            reward_ext,
            reward_int,
            int(not worker.terminated),
            prev_onehot_action,
            prev_reward_ext,
            prev_reward_int,
            self.actor_index,
        ]

        if not self.distributed:
            priority = None
        elif not self.config.requires_priority():
            priority = None
        else:
            next_invalid_actions_idx = [0 for _ in next_invalid_actions]
            _params = [
                next_state[np.newaxis, ...],
                np.array([[prev_reward_ext]], dtype=np.float32),
                np.array([[prev_reward_int]], dtype=np.float32),
                self.prev_onehot_action[np.newaxis, ...],
                self.onehot_actor_idx,
                next_invalid_actions_idx,
                next_invalid_actions,
                np.array([int(not worker.terminated)], dtype=np.float32),
                np.array([self.discount], dtype=np.float32),
            ]
            target_q_ext = self.parameter.calc_target_q(
                True,
                np.array([reward_ext], dtype=np.float32),
                *_params,
            )[0]

            if self.config.disable_int_priority or not self.config.enable_intrinsic_reward:
                priority = abs(target_q_ext - self.q_ext[worker.action])
            elif self.beta == 0:
                priority = abs(target_q_ext - self.q_ext[worker.action])
            else:
                target_q_int = self.parameter.calc_target_q(
                    False,
                    np.array([reward_int], dtype=np.float32),
                    *_params,
                )[0]

                priority = abs((target_q_ext + self.beta * target_q_int) - self.q[worker.action])

        self.memory.add(batch, priority)
        return

    def _calc_episodic_reward(self, state):
        k = self.config.episodic_count_max
        epsilon = self.config.episodic_epsilon
        cluster_distance = self.config.episodic_cluster_distance
        c = self.config.episodic_pseudo_counts

        # 埋め込み関数から制御可能状態を取得
        cont_state = self.parameter.predict_emb(state)[0]

        # 初回
        if len(self.episodic_memory) == 0:
            self.episodic_memory.append(cont_state)
            return 1 / c

        # エピソードメモリ内の全要素とユークリッド距離を求める
        euclidean_list = [np.linalg.norm(m - cont_state, ord=2) for m in self.episodic_memory]

        # エピソードメモリに制御可能状態を追加
        self.episodic_memory.append(cont_state)

        # 近いk個を対象
        euclidean_list = np.sort(euclidean_list)[:k]

        # 上位k個の移動平均を出す
        mode_ave = np.mean(euclidean_list)
        if mode_ave == 0.0:
            # ユークリッド距離は正なので平均0は全要素0のみ
            dn = euclidean_list
        else:
            dn = euclidean_list / mode_ave  # 正規化

        # 一定距離以下を同じ状態とする
        dn = np.maximum(dn - cluster_distance, 0)

        # 訪問回数を計算(Dirac delta function の近似)
        dn = epsilon / (dn + epsilon)
        N = np.sum(dn)

        # 報酬の計算
        reward = 1 / (np.sqrt(N) + c)
        return reward

    def _calc_lifelong_reward(self, state):
        # RND取得
        rnd_target_val = self.parameter.predict_lifelong_target(state)[0]
        rnd_train_val = self.parameter.predict_lifelong_train(state)[0]

        # MSE
        error = np.square(rnd_target_val - rnd_train_val).mean()
        reward = 1 + error

        if reward < 1:
            reward = 1
        if reward > self.config.lifelong_max:
            reward = self.config.lifelong_max

        return reward

    def render_terminal(self, worker, **kwargs) -> None:
        if self.config.enable_rescale:
            q = funcs.inverse_rescaling(self.q)
            q_ext = funcs.inverse_rescaling(self.q_ext)
            q_int = funcs.inverse_rescaling(self.q_int)
        else:
            q = self.q
            q_ext = self.q_ext
            q_int = self.q_int
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:6.3f} = {q_ext[a]:6.3f} + {self.beta:.3f} * {q_int[a]:6.3f}"

        funcs.render_discrete_action(int(maxa), self.config.action_space, worker.env, _render_sub)
