import collections
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, cast

import numpy as np

from srl.base.define import EnvObservationTypes, RLTypes
from srl.base.rl.algorithms.discrete_action import DiscreteActionWorker
from srl.base.rl.base import RLParameter
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.rl.functions.common import (
    calc_epsilon_greedy_probs,
    create_beta_list,
    create_discount_list,
    create_epsilon_list,
    inverse_rescaling,
    random_choice_by_probs,
    render_discrete_action,
    rescaling,
)
from srl.rl.memories.priority_experience_replay import PriorityExperienceReplay, PriorityExperienceReplayConfig
from srl.rl.models.dueling_network import DuelingNetworkConfig
from srl.rl.models.framework_config import FrameworkConfig
from srl.rl.models.image_block import ImageBlockConfig
from srl.rl.models.mlp_block import MLPBlockConfig
from srl.rl.processors.image_processor import ImageProcessor
from srl.rl.schedulers.scheduler import SchedulerConfig

"""
Paper: https://arxiv.org/abs/2003.13350

DQN
    window_length          : x
    Fixed Target Q-Network : o
    Error clipping     : o
    Experience Replay  : o
    Frame skip         : -
    Annealing e-greedy : x
    Reward clip        : x
    Image preprocessor : -
Rainbow
    Double DQN               : o
    Priority Experience Reply: o
    Dueling Network          : o
    Multi-Step learning      : x
    Noisy Network            : x
    Categorical DQN          : x
Recurrent Replay Distributed DQN(R2D2)
    LSTM                     : o
    Value function rescaling : o
Never Give Up(NGU)
    Intrinsic Reward : o
    UVFA             : o
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
    """<:ref:`PriorityExperienceReplay`>"""

    #: ε-greedy parameter for Test
    test_epsilon: float = 0
    #: intrinsic reward rate for Test
    test_beta: float = 0

    #: <:ref:`Framework`>
    framework: FrameworkConfig = field(init=False, default_factory=lambda: FrameworkConfig())
    #: <:ref:`ImageBlock`> This layer is only used when the input is an image.
    image_block: ImageBlockConfig = field(init=False, default_factory=lambda: ImageBlockConfig())
    #:  Lstm units
    lstm_units: int = 512
    #: <:ref:`DuelingNetwork`> hidden layer
    dueling_network: DuelingNetworkConfig = field(init=False, default_factory=lambda: DuelingNetworkConfig())

    #: <:ref:`scheduler`> Learning rate
    lr_ext: float = 0.0001  # type: ignore , type OK
    #: <:ref:`scheduler`> Intrinsic network Learning rate
    lr_int: float = 0.0001  # type: ignore , type OK
    #: Synchronization interval to Target network
    target_model_update_interval: int = 1500

    #: Burn-in steps
    burnin: int = 5
    #: LSTM input length
    sequence_length: int = 5
    #: retrace parameter h
    retrace_h: float = 1.0

    #: enable DoubleDQN
    enable_double_dqn: bool = True
    #: enable rescaling
    enable_rescale: bool = False

    #: [sliding-window UCB] actor num
    actor_num: int = 32
    #: [sliding-window UCB] UCBで使う直近のエピソード数
    ucb_window_size: int = 3600
    #: [sliding-window UCB] UCBを使う確率
    ucb_epsilon: float = 0.01
    #: [sliding-window UCB] UCB β
    ucb_beta: float = 1

    #: enable intrinsic reward
    enable_intrinsic_reward: bool = True

    #: <:ref:`scheduler`> Episodic Learning rate
    episodic_lr: float = 0.0005  # type: ignore , type OK
    episodic_count_max: int = 10  # k
    episodic_epsilon: float = 0.001
    episodic_cluster_distance: float = 0.008
    episodic_memory_capacity: int = 30000
    episodic_pseudo_counts: float = 0.1  # 疑似カウント定数
    episodic_emb_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())
    episodic_out_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())

    #: <:ref:`scheduler`> Lifelong Learning rate
    lifelong_lr: float = 0.0005  # type: ignore , type OK
    lifelong_max: float = 5.0  # L
    lifelong_hidden_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())

    # UVFA
    input_ext_reward: bool = True
    input_int_reward: bool = False
    input_action: bool = False

    # other
    disable_int_priority: bool = False  # Not use internal rewards to calculate priority
    dummy_state_val: float = 0.0

    def __post_init__(self):
        super().__post_init__()

        self.lr_ext: SchedulerConfig = SchedulerConfig(cast(float, self.lr_ext))
        self.lr_int: SchedulerConfig = SchedulerConfig(cast(float, self.lr_int))
        self.episodic_lr: SchedulerConfig = SchedulerConfig(cast(float, self.episodic_lr))
        self.lifelong_lr: SchedulerConfig = SchedulerConfig(cast(float, self.lifelong_lr))
        self.memory.set_proportional_memory()

        self.dueling_network.set((512,), True)
        self.episodic_emb_block.set_mlp(
            (32,),
            activation="relu",
            # kernel_initializer="he_normal",
            # dense_kwargs={"bias_initializer": keras.initializers.constant(0.001)},
        )
        self.episodic_out_block.set_mlp((128,))
        self.lifelong_hidden_block.set_mlp((128,))

    def set_atari_config(self):
        """Set the Atari parameters written in the paper."""
        self.lr_ext.set_constant(0.0001)
        self.lr_int.set_constant(0.0001)
        self.lifelong_lr.set_constant(0.0005)
        self.episodic_lr.set_constant(0.0005)
        self.batch_size = 64
        self.lstm_units = 512
        self.image_block.set_dqn_image()
        self.dueling_network.set((512,), enable=True)
        self.discount = 0.99

        self.burnin = 40
        self.sequence_length = 80
        self.retrace_h = 0.95

        self.episodic_memory_capacity = 30_000

        self.memory.set_replay_memory()
        self.memory.capacity = 100_000
        self.memory.warmup_size = 6250
        self.image_block.set_dqn_image()

        self.target_model_update_interval = 1500

    def set_processor(self) -> List[Processor]:
        return [
            ImageProcessor(
                image_type=EnvObservationTypes.GRAY_2ch,
                resize=(84, 84),
                enable_norm=True,
            )
        ]

    @property
    def base_action_type(self) -> RLTypes:
        return RLTypes.DISCRETE

    @property
    def base_observation_type(self) -> RLTypes:
        return RLTypes.CONTINUOUS

    def get_use_framework(self) -> str:
        return self.framework.get_use_framework()

    def getName(self) -> str:
        return f"Agent57:{self.get_use_framework()}"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_memory()
        assert self.burnin >= 0
        assert self.sequence_length >= 1


# ------------------------------------------------------
# Memory
# ------------------------------------------------------
class Memory(PriorityExperienceReplay):
    pass


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class CommonInterfaceParameter(RLParameter, ABC):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.beta_list = create_beta_list(self.config.actor_num)
        self.discount_list = create_discount_list(self.config.actor_num)
        self.epsilon_list = create_epsilon_list(self.config.actor_num)

    @abstractmethod
    def get_initial_hidden_state_q_ext(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def get_initial_hidden_state_q_int(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def predict_q_ext_online(self, x, hidden_state) -> Tuple[np.ndarray, Any]:
        raise NotImplementedError()

    @abstractmethod
    def predict_q_int_online(self, x, hidden_state) -> Tuple[np.ndarray, Any]:
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

    @abstractmethod
    def convert_numpy_from_hidden_state(self, h):
        raise NotImplementedError()

    def change_batchs_format(self, batchs, weights):
        (
            states,
            onehot_actions,
            # probs,  # [1]
            rewards_ext,
            rewards_int,
            dones,
            actors,
            _,  # invalid_actions
            hidden_states_ext,
            hidden_states_int,
        ) = zip(*batchs)
        # (batch, step, x)
        states = np.asarray(states)

        # (batch, step, 1)
        rewards_ext = np.array(rewards_ext, dtype=np.float32)[..., np.newaxis]
        rewards_int = np.array(rewards_int, dtype=np.float32)[..., np.newaxis]

        burnin_states = states[:, : self.config.burnin, ...]
        burnin_rewards_ext = rewards_ext[:, : self.config.burnin, ...]
        burnin_rewards_int = rewards_int[:, : self.config.burnin, ...]
        instep_states = states[:, self.config.burnin :, ...]
        instep_rewards_ext = rewards_ext[:, self.config.burnin :, ...]
        instep_rewards_int = rewards_int[:, self.config.burnin :, ...]

        # (batch, step, action)
        onehot_actions = np.asarray(onehot_actions, dtype=np.float32)
        burnin_actions_onehot = onehot_actions[:, : self.config.burnin, ...]
        instep_actions_onehot = onehot_actions[:, self.config.burnin :, ...]

        step_actions_onehot = instep_actions_onehot[:, 1:, ...]
        step_rewards_ext = instep_rewards_ext[:, 1:].squeeze(axis=2)  # (batch, step)
        step_rewards_int = instep_rewards_int[:, 1:].squeeze(axis=2)  # (batch, step)
        step_dones = np.array(dones, dtype=np.float32)  # (batch, step)

        # (batch)
        discounts = np.array([self.discount_list[a] for a in actors], dtype=np.float32)
        # (batch)
        beta_list = np.array([self.beta_list[a] for a in actors], dtype=np.float32)
        actors = np.asarray(actors)

        # ファンシーインデックス
        inv_act_idx1 = [i for i, b in enumerate(batchs) for e in b[6] for e2 in e]
        inv_act_idx2 = [i for b in batchs for i, e in enumerate(b[6]) for e2 in e]
        inv_act_idx3 = [e2 for b in batchs for e in b[6] for e2 in e]

        # (batch, step, action)
        burnin_actor_onehot = np.tile(actors[:, np.newaxis], (1, self.config.burnin))
        burnin_actor_onehot = np.identity(self.config.actor_num, dtype=np.float32)[burnin_actor_onehot]
        instep_actor_onehot = np.tile(actors[:, np.newaxis], (1, self.config.sequence_length + 1))
        instep_actor_onehot = np.identity(self.config.actor_num, dtype=np.float32)[instep_actor_onehot]

        # (step, batch)
        weights = np.tile(weights[np.newaxis, :], (self.config.sequence_length, 1))

        return (
            burnin_states,
            burnin_rewards_ext,
            burnin_rewards_int,
            burnin_actions_onehot,
            burnin_actor_onehot,
            instep_states,
            instep_rewards_ext,
            instep_rewards_int,
            instep_actions_onehot,
            instep_actor_onehot,
            step_rewards_ext,
            step_rewards_int,
            step_actions_onehot,
            step_dones,
            inv_act_idx1,
            inv_act_idx2,
            inv_act_idx3,
            hidden_states_ext,
            hidden_states_int,
            discounts,
            beta_list,
            weights,
        )

    def calc_target_q(
        self,
        q: np.ndarray,  # (batch, step+1, action), next_q含む
        q_target: np.ndarray,  # (batch, step+1, action), next_q含む
        action_q: np.ndarray,  # (batch, step)
        #
        step_rewards,  # (batch, step)
        step_actions_onehot,  # (batch, step, action)
        step_dones,  # (batch, step)
        inv_act_idx1,
        inv_act_idx2,
        inv_act_idx3,
        discounts,  # (batch)
    ):
        n_q = q[:, 1:, :]
        n_q_target = q_target[:, 1:, :]

        # --- calc TD error
        # ファンシーインデックス
        if self.config.enable_double_dqn:
            n_q[inv_act_idx1, inv_act_idx2, inv_act_idx3] = -np.inf
            n_act_idx = np.argmax(n_q, axis=2)
        else:
            n_q_target[inv_act_idx1, inv_act_idx2, inv_act_idx3] = -np.inf
            n_act_idx = np.argmax(n_q_target, axis=2)
        maxq = np.take_along_axis(n_q_target, np.expand_dims(n_act_idx, axis=2), axis=2)
        maxq = np.squeeze(maxq, axis=2)

        if self.config.enable_rescale:
            maxq = inverse_rescaling(maxq)

        # (batch) -> (batch, step)
        discounts_repeat = np.repeat(discounts, self.config.sequence_length).reshape(
            (self.config.batch_size, self.config.sequence_length)
        )
        gains = step_rewards + step_dones * discounts_repeat * maxq

        if self.config.enable_rescale:
            gains = rescaling(gains)

        # (batch, step, shape) ->  (step, batch, shape)
        gains = np.swapaxes(gains, 1, 0)
        # (batch, step) -> (step, batch)
        action_q = np.swapaxes(action_q, 1, 0)

        # --- calc retrace
        # 各batchで最大のアクションを選んでるかどうか
        # greedyな方策なので、最大アクションなら確率1.0 #[1]
        pi_probs = np.argmax(step_actions_onehot, axis=2) == n_act_idx

        #  (batch, seq, shape) -> (seq, batch, shape)
        pi_probs = np.transpose(pi_probs, (1, 0))

        retrace_seq = [np.ones((self.config.batch_size,), dtype=np.float32)]  # 0stepはretraceなし
        retrace = np.ones((self.config.batch_size,), dtype=np.float32)
        discounts_seq = [discounts]
        for t in range(self.config.sequence_length - 1):
            # #[1]
            # pi_probs は 0 or 1 で mu_probs は1以下なので必ず 0 or 1 になる
            # retrace *= self.config.retrace_h * np.minimum(1, pi_probs[n] / mu_probs[n])
            retrace *= self.config.retrace_h * pi_probs[t]
            retrace_seq.append(retrace.copy())
            discounts_seq.append(discounts_seq[-1] * discounts)
        retrace_seq = np.asarray(retrace_seq)
        discounts_seq = np.asarray(discounts_seq)

        # --- calc td error 後ろから計算
        target_q_list = []
        next_td_error = 0
        for t in reversed(range(self.config.sequence_length)):
            target_q = gains[t] + retrace_seq[t] * discounts_seq[t] * next_td_error
            target_q_list.insert(0, target_q)
            next_td_error = target_q - action_q[t]
        target_q_list = np.asarray(target_q_list)

        return target_q_list


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: CommonInterfaceParameter = self.parameter

        self.dummy_state = np.full(self.config.observation_shape, self.config.dummy_state_val, dtype=np.float32)
        self.act_onehot_arr = np.identity(self.config.action_num, dtype=int)

        # actor
        self.beta_list = create_beta_list(self.config.actor_num)
        self.epsilon_list = create_epsilon_list(self.config.actor_num)
        self.discount_list = create_discount_list(self.config.actor_num)

        # ucb
        self.actor_index = -1
        self.ucb_recent = []
        self.ucb_actors_count = [1 for _ in range(self.config.actor_num)]  # 1回は保証
        self.ucb_actors_reward = [0.0 for _ in range(self.config.actor_num)]

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        self.q_ext = [0] * self.config.action_num
        self.q_int = [0] * self.config.action_num
        self.q = [0] * self.config.action_num
        self.episodic_reward = 0
        self.lifelong_reward = 0
        self.reward_int = 0

        # states : burnin + sequence_length + next_state
        # actions: burnin + sequence_length + prev_action
        # probs  : sequence_length
        # rewards: burnin + sequence_length + prev_reward
        # done   : sequence_length
        # invalid_actions: sequence_length + next_invalid_actions
        # hidden_state   : burnin + sequence_length + next_state

        self.recent_states = [self.dummy_state for _ in range(self.config.burnin + self.config.sequence_length + 1)]
        self.recent_actions = [
            self.act_onehot_arr[random.randint(0, self.config.action_num - 1)]
            for _ in range(self.config.burnin + self.config.sequence_length + 1)
        ]
        # self.recent_probs = [1.0 / self.config.action_num for _ in range(self.config.sequence_length)] #[1]
        self.recent_rewards_ext = [0.0 for _ in range(self.config.burnin + self.config.sequence_length + 1)]
        self.recent_rewards_int = [0.0 for _ in range(self.config.burnin + self.config.sequence_length + 1)]
        self.recent_done = [1 for _ in range(self.config.sequence_length)]
        self.recent_next_invalid_actions = [[] for _ in range(self.config.sequence_length)]

        self.hidden_state_ext = self.parameter.get_initial_hidden_state_q_ext()
        self.hidden_state_int = self.parameter.get_initial_hidden_state_q_int()
        self.recent_hidden_states_ext = [
            self.parameter.convert_numpy_from_hidden_state(self.hidden_state_ext)
            for _ in range(self.config.burnin + self.config.sequence_length + 1)
        ]
        self.recent_hidden_states_int = [
            self.parameter.convert_numpy_from_hidden_state(self.hidden_state_int)
            for _ in range(self.config.burnin + self.config.sequence_length + 1)
        ]

        self.recent_states.pop(0)
        self.recent_states.append(state.astype(np.float32))

        # TD誤差を計算するか
        if not self.distributed:
            self._calc_td_error = False
        elif not self.config.memory.requires_priority():
            self._calc_td_error = False
        else:
            self._calc_td_error = True
            self._history_batch = []

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

        self.action = random.randint(0, self.config.action_num - 1)
        self.reward_ext = 0
        self.reward_int = 0

        # Q値取得用
        self.onehot_actor_idx = np.identity(self.config.actor_num, dtype=np.float32)[self.actor_index][
            np.newaxis, np.newaxis, ...
        ]

        # sliding-window UCB 用に報酬を保存
        self.episode_reward = 0.0

        # エピソードメモリ(エピソード毎に初期化)
        self.episodic_memory = collections.deque(maxlen=self.config.episodic_memory_capacity)

        return {}

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
        return np.random.choice(np.where(ucbs == np.max(ucbs))[0])

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        prev_onehot_action = np.identity(self.config.action_num, dtype=np.float32)[self.action][
            np.newaxis, np.newaxis, ...
        ]

        in_ = [
            self.recent_states[-1][np.newaxis, np.newaxis, ...],
            np.array([[[self.reward_ext]]], dtype=np.float32),
            np.array([[[self.reward_int]]], dtype=np.float32),
            prev_onehot_action,
            self.onehot_actor_idx,
        ]
        self.q_ext, self.hidden_state_ext = self.parameter.predict_q_ext_online(in_, self.hidden_state_ext)
        self.q_int, self.hidden_state_int = self.parameter.predict_q_int_online(in_, self.hidden_state_int)
        self.q_ext = self.q_ext[0][0]
        self.q_int = self.q_int[0][0]
        self.q = self.q_ext + self.beta * self.q_int

        probs = calc_epsilon_greedy_probs(self.q, invalid_actions, self.epsilon, self.config.action_num)
        self.action = random_choice_by_probs(probs)
        # self.prob = probs[self.action]  #[1]
        return self.action, {}

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward_ext: float,
        done: bool,
        next_invalid_actions: List[int],
    ) -> Dict:
        self.episode_reward += reward_ext
        self.reward_ext = reward_ext

        # 内部報酬
        if self.config.enable_intrinsic_reward:
            n_s = next_state[np.newaxis, ...]
            self.episodic_reward = self._calc_episodic_reward(n_s)
            self.lifelong_reward = self._calc_lifelong_reward(n_s)
            self.reward_int = self.episodic_reward * self.lifelong_reward

            _info = {
                "episodic": self.episodic_reward,
                "lifelong": self.lifelong_reward,
                "reward_int": self.reward_int,
            }
        else:
            self.reward_int = 0.0
            _info = {}

        self.recent_states.pop(0)
        self.recent_states.append(next_state)
        self.recent_actions.pop(0)
        self.recent_actions.append(self.act_onehot_arr[self.action])
        # self.recent_probs.pop(0)
        # self.recent_probs.append(self.prob)  #[1]
        self.recent_rewards_ext.pop(0)
        self.recent_rewards_ext.append(reward_ext)
        self.recent_rewards_int.pop(0)
        self.recent_rewards_int.append(self.reward_int)
        self.recent_done.pop(0)
        self.recent_done.append(0 if done else 1)
        self.recent_next_invalid_actions.pop(0)
        self.recent_next_invalid_actions.append(next_invalid_actions)
        self.recent_hidden_states_ext.pop(0)
        self.recent_hidden_states_ext.append(self.parameter.convert_numpy_from_hidden_state(self.hidden_state_ext))
        self.recent_hidden_states_int.pop(0)
        self.recent_hidden_states_int.append(self.parameter.convert_numpy_from_hidden_state(self.hidden_state_int))

        if not self.training:
            return _info

        if self._calc_td_error:
            calc_info = {
                "q": self.q[self.action],
                "reward_ext": reward_ext,
                "reward_int": self.reward_int,
            }
        else:
            calc_info = None

        self._add_memory(calc_info)

        if done:
            # 残りstepも追加
            for _ in range(len(self.recent_rewards_ext) - 1):
                self.recent_states.pop(0)
                self.recent_states.append(self.dummy_state)
                self.recent_actions.pop(0)
                self.recent_actions.append(self.act_onehot_arr[random.randint(0, self.config.action_num - 1)])
                # self.recent_probs.pop(0)
                # self.recent_probs.append(1.0 / self.config.action_num) #[1]
                self.recent_rewards_ext.pop(0)
                self.recent_rewards_ext.append(0.0)
                self.recent_rewards_int.pop(0)
                self.recent_rewards_int.append(0.0)
                self.recent_done.pop(0)
                self.recent_done.append(0)
                self.recent_next_invalid_actions.pop(0)
                self.recent_next_invalid_actions.append([])
                self.recent_hidden_states_ext.pop(0)
                self.recent_hidden_states_int.pop(0)

                self._add_memory(
                    {
                        "q": self.q[self.action],
                        "reward_ext": 0.0,
                        "reward_int": 0.0,
                    }
                )

                if self._calc_td_error:
                    # TD誤差を計算してメモリに送る
                    # targetQはモンテカルロ法
                    reward_ext = 0
                    reward_int = 0
                    for batch, info in reversed(self._history_batch):
                        if self.config.enable_rescale:
                            _r_ext = inverse_rescaling(reward_ext)
                            _r_int = inverse_rescaling(reward_int)
                        else:
                            _r_ext = reward_ext
                            _r_int = reward_int
                        reward_ext = info["reward_ext"] + self.discount * _r_ext
                        reward_int = info["reward_int"] + self.discount * _r_int
                        if self.config.enable_rescale:
                            reward_ext = rescaling(reward_ext)
                            reward_int = rescaling(reward_int)

                        if self.config.disable_int_priority:
                            priority = abs(reward_ext - info["q"])
                        else:
                            priority = abs((reward_ext + self.beta * reward_int) - info["q"])
                        self.memory.add(batch, priority)

        return _info

    def _add_memory(self, calc_info):
        """
        [
            states,
            onehot_actions,
            # probs,
            rewards_ext,
            rewards_int,
            dones,
            actor,
            invalid_actions,
            hidden_states_ext,
            hidden_states_int,
        ]
        """
        batch = [
            self.recent_states[:],
            self.recent_actions[:],
            # self.recent_probs[:],  #[1]
            self.recent_rewards_ext[:],
            self.recent_rewards_int[:],
            self.recent_done[:],
            self.actor_index,
            self.recent_next_invalid_actions[:],
            self.recent_hidden_states_ext[0],
            self.recent_hidden_states_int[0],
        ]

        if self._calc_td_error:
            # エピソード最後に計算してメモリに送る
            self._history_batch.append([batch, calc_info])
        else:
            # 計算する必要がない場合はそのままメモリに送る
            self.memory.add(batch, None)

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
            q = inverse_rescaling(self.q)
            q_ext = inverse_rescaling(self.q_ext)
            q_int = inverse_rescaling(self.q_int)
        else:
            q = self.q
            q_ext = self.q_ext
            q_int = self.q_int

        print(f"episodic_reward: {self.episodic_reward}")
        print(f"lifelong_reward: {self.lifelong_reward}")
        print(f"reward_int     : {self.reward_int}")

        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            s = f"{q[a]:5.3f}"
            s += f"{a:2d}: {q[a]:5.3f} = {q_ext[a]:5.3f} + {self.beta} * {q_int[a]:5.3f}"
            return s

        render_discrete_action(maxa, worker.env, self.config, _render_sub)
