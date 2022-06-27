import collections
import logging
import random
from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from srl.base.define import RLObservationType
from srl.base.env.base import EnvRun
from srl.base.rl.algorithms.discrete_action import (DiscreteActionConfig,
                                                    DiscreteActionWorker)
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import PriorityExperienceReplay
from srl.rl.functions.common import (create_beta_list, create_epsilon_list,
                                     create_gamma_list, inverse_rescaling,
                                     render_discrete_action, rescaling)
from srl.rl.functions.dueling_network import create_dueling_network_layers
from srl.rl.functions.model import ImageLayerType, create_input_layers
from tensorflow.keras import layers as kl

logger = logging.getLogger(__name__)

"""
DQN
    window_length          : o
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
class Config(DiscreteActionConfig):

    # test
    test_epsilon: float = 0
    test_beta: float = 0

    # model
    window_length: int = 1
    hidden_layer_sizes: Tuple[int, ...] = (512,)
    activation: str = "relu"
    image_layer_type: ImageLayerType = ImageLayerType.DQN
    batch_size: int = 32
    q_ext_lr: float = 0.001
    q_int_lr: float = 0.001
    target_model_update_interval: int = 1000

    # rescale
    enable_rescale: bool = False

    # double dqn
    enable_double_dqn: bool = True

    # DuelingNetwork
    enable_dueling_network: bool = True
    dueling_network_type: str = "average"

    # Priority Experience Replay
    capacity: int = 100_000
    memory_name: str = "ProportionalMemory"
    memory_warmup_size: int = 1000
    memory_alpha: float = 0.6
    memory_beta_initial: float = 1.0
    memory_beta_steps: int = 1_000_000

    # ucb(160,0.5 or 3600,0.01)
    actor_num: int = 32
    ucb_window_size: int = 3600  # UCB上限
    ucb_epsilon: float = 0.01  # UCBを使う確率
    ucb_beta: float = 1  # UCBのβ

    # intrinsic reward
    enable_intrinsic_reward: bool = True

    # episodic
    episodic_lr: float = 0.0005
    episodic_count_max: int = 10  # k
    episodic_epsilon: float = 0.001
    episodic_cluster_distance: float = 0.008
    episodic_memory_capacity: int = 30000
    episodic_pseudo_counts: float = 0.1  # 疑似カウント定数
    episodic_hidden_layer_sizes1: Tuple[int, ...] = (32,)
    episodic_hidden_layer_sizes2: Tuple[int, ...] = (128,)

    # lifelong
    lifelong_lr: float = 0.00001
    lifelong_max: float = 5.0  # L
    lifelong_hidden_layer_sizes: Tuple[int, ...] = (128,)

    # UVFA
    input_ext_reward: bool = True
    input_int_reward: bool = False
    input_action: bool = False

    # other
    disable_int_priority: bool = False  # Not use internal rewards to calculate priority

    dummy_state_val: float = 0.0

    def __post_init__(self):
        super().__init__()

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.CONTINUOUS

    @staticmethod
    def getName() -> str:
        return "Agent57_light"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.window_length > 0
        assert self.memory_warmup_size < self.capacity
        assert self.batch_size < self.memory_warmup_size
        assert len(self.hidden_layer_sizes) > 0


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


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()
        self.input_ext_reward = config.input_ext_reward
        self.input_int_reward = config.input_int_reward
        self.input_action = config.input_action
        if not config.enable_intrinsic_reward:
            self.input_int_reward = False

        in_state, c = create_input_layers(
            config.window_length,
            config.observation_shape,
            config.env_observation_type,
            config.image_layer_type,
        )

        # UVFA
        input_list = []
        if self.input_ext_reward:
            input_list.append(kl.Input(shape=(1,)))
        if self.input_int_reward:
            input_list.append(kl.Input(shape=(1,)))
        if self.input_action:
            input_list.append(kl.Input(shape=(config.action_num,)))
        input_list.append(kl.Input(shape=(config.actor_num,)))
        c = kl.Concatenate()([c] + input_list)

        for i in range(len(config.hidden_layer_sizes) - 1):
            c = kl.Dense(
                config.hidden_layer_sizes[i],
                activation=config.activation,
                kernel_initializer="he_normal",
            )(c)

        if config.enable_dueling_network:
            c = create_dueling_network_layers(
                c,
                config.action_num,
                config.hidden_layer_sizes[-1],
                config.dueling_network_type,
                activation=config.activation,
            )
        else:
            c = kl.Dense(config.hidden_layer_sizes[-1], activation=config.activation, kernel_initializer="he_normal")(
                c
            )
            c = kl.Dense(
                config.action_num, kernel_initializer="truncated_normal", bias_initializer="truncated_normal"
            )(c)

        self.model = keras.Model([in_state] + input_list, c)

        # 重みを初期化
        dummy1 = np.zeros(shape=(1, config.window_length) + config.observation_shape, dtype=float)
        dummy2 = np.zeros(shape=(1, 1), dtype=float)
        dummy3 = np.zeros(shape=(1, 1), dtype=float)
        dummy4 = np.zeros(shape=(1, config.action_num), dtype=float)
        dummy5 = np.zeros(shape=(1, config.actor_num), dtype=float)
        val = self(dummy1, dummy2, dummy3, dummy4, dummy5)
        assert val.shape == (1, config.action_num)

    def call(self, state, reward_ext, reward_int, onehot_action, onehot_actor):
        input_list = [state]
        if self.input_ext_reward:
            input_list.append(reward_ext)
        if self.input_int_reward:
            input_list.append(reward_int)
        if self.input_action:
            input_list.append(onehot_action)
        input_list.append(onehot_actor)
        return self.model(input_list)


# ------------------------------------------------------
# エピソード記憶部(episodic_reward)
# ------------------------------------------------------
class _EmbeddingNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        # in model
        in_state, c = create_input_layers(
            config.window_length,
            config.observation_shape,
            config.env_observation_type,
            config.image_layer_type,
        )
        for h in config.episodic_hidden_layer_sizes1:
            c = kl.Dense(
                h,
                activation="relu",
                kernel_initializer="he_normal",
                bias_initializer=keras.initializers.constant(0.001),
            )(c)
        self.model1 = keras.Model(in_state, c)

        # out model
        out_h = config.episodic_hidden_layer_sizes1[-1]
        in1 = kl.Input(shape=(out_h,))
        in2 = kl.Input(shape=(out_h,))
        c = kl.Concatenate()([in1, in2])
        for h in config.episodic_hidden_layer_sizes2:
            c = kl.Dense(
                h,
                activation="relu",
                kernel_initializer="he_normal",
            )(c)
        c = kl.LayerNormalization()(c)
        c = kl.Dense(config.action_num, activation="softmax")(c)
        self.model2 = keras.Model([in1, in2], c)

        # 重みを初期化
        dummy_state = np.zeros(shape=(1, config.window_length) + config.observation_shape, dtype=np.float32)
        val = self(dummy_state, dummy_state)
        assert val.shape == (1, config.action_num)

    def call(self, state1, state2):
        c1 = self.model1(state1)
        c2 = self.model1(state2)
        return self.model2([c1, c2])

    def predict(self, state):
        return self.model1(state)


# ------------------------------------------------------
# 生涯記憶部(life long novelty module)
# ------------------------------------------------------
class _LifelongNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        in_state, c = create_input_layers(
            config.window_length,
            config.observation_shape,
            config.env_observation_type,
            config.image_layer_type,
        )

        for h in config.lifelong_hidden_layer_sizes:
            c = kl.Dense(
                h,
                activation="relu",
                kernel_initializer="he_normal",
                bias_initializer="he_normal",
            )(c)
        c = kl.LayerNormalization()(c)
        self.model = keras.Model(in_state, c)

        # 重みを初期化
        dummy_state = np.zeros(shape=(1, config.window_length) + config.observation_shape, dtype=np.float32)
        val = self(dummy_state)
        assert val.shape == (1, config.lifelong_hidden_layer_sizes[-1])

    def call(self, state):
        return self.model(state)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.q_ext_online = _QNetwork(self.config)
        self.q_ext_target = _QNetwork(self.config)
        self.q_int_online = _QNetwork(self.config)
        self.q_int_target = _QNetwork(self.config)
        self.emb_network = _EmbeddingNetwork(self.config)
        self.lifelong_target = _LifelongNetwork(self.config)
        self.lifelong_train = _LifelongNetwork(self.config)

    def restore(self, data: Any) -> None:
        self.q_ext_online.set_weights(data[0])
        self.q_ext_target.set_weights(data[0])
        self.q_int_online.set_weights(data[1])
        self.q_int_target.set_weights(data[1])
        self.emb_network.set_weights(data[2])
        self.lifelong_target.set_weights(data[3])
        self.lifelong_train.set_weights(data[4])

    def backup(self):
        d = [
            self.q_ext_online.get_weights(),
            self.q_int_online.get_weights(),
            self.emb_network.get_weights(),
            self.lifelong_target.get_weights(),
            self.lifelong_train.get_weights(),
        ]
        return d

    def summary(self):
        self.q_ext_online.model.summary()
        self.emb_network.model1.summary()
        self.emb_network.model2.summary()
        self.lifelong_target.model.summary()

    # ---------------------------------

    def calc_target_q(
        self,
        q_online,
        q_target,
        rewards,
        #
        n_states,
        actions_onehot,
        next_invalid_actions_list,
        rewards_ext,
        rewards_int,
        done_list,
        actor_idx_onehot,
        gamma_list,
    ):
        n_q = q_online(n_states, rewards_ext, rewards_int, actions_onehot, actor_idx_onehot).numpy()
        n_q_target = q_target(n_states, rewards_ext, rewards_int, actions_onehot, actor_idx_onehot).numpy()

        target_q = np.zeros(len(rewards))
        for i in range(len(rewards)):
            if done_list[i]:
                gain = rewards[i]
            else:
                # DoubleDQN: indexはonlineQから選び、値はtargetQを選ぶ
                next_invalid_actions = next_invalid_actions_list[i]
                if self.config.enable_double_dqn:
                    n_q[i] = [(-np.inf if a in next_invalid_actions else v) for a, v in enumerate(n_q[i])]
                    n_act_idx = np.argmax(n_q[i])
                else:
                    n_q_target[i] = [
                        (-np.inf if a in next_invalid_actions else v) for a, v in enumerate(n_q_target[i])
                    ]
                    n_act_idx = np.argmax(n_q_target[i])
                maxq = n_q_target[i][n_act_idx]
                if self.config.enable_rescale:
                    maxq = inverse_rescaling(maxq)
                gain = rewards[i] + gamma_list[i] * maxq
            if self.config.enable_rescale:
                gain = rescaling(gain)
            target_q[i] = gain

        return target_q


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        self.q_ext_optimizer = keras.optimizers.Adam(learning_rate=self.config.q_ext_lr)
        self.q_int_optimizer = keras.optimizers.Adam(learning_rate=self.config.q_int_lr)
        self.q_loss = keras.losses.Huber()

        self.emb_optimizer = keras.optimizers.Adam(learning_rate=self.config.episodic_lr)
        self.emb_loss = keras.losses.MeanSquaredError()

        self.lifelong_optimizer = keras.optimizers.Adam(learning_rate=self.config.lifelong_lr)
        self.lifelong_loss = keras.losses.MeanSquaredError()

        self.beta_list = create_beta_list(self.config.actor_num)
        self.gamma_list = create_gamma_list(self.config.actor_num)
        self.epsilon_list = create_epsilon_list(self.config.actor_num)

        self.train_count = 0
        self.sync_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):

        if self.remote_memory.length() < self.config.memory_warmup_size:
            return {}

        indices, batchs, weights = self.remote_memory.sample(self.train_count, self.config.batch_size)
        td_errors, info = self._train_on_batchs(batchs, weights)
        self.remote_memory.update(indices, batchs, td_errors)

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_ext_target.set_weights(self.parameter.q_ext_online.get_weights())
            self.parameter.q_int_target.set_weights(self.parameter.q_int_online.get_weights())
            self.sync_count += 1

        self.train_count += 1
        info["sync"] = self.sync_count
        return info

    def _train_on_batchs(self, batchs, weights):

        # データ形式を変形
        states = []
        n_states = []
        actions = []
        next_invalid_actions_list = []
        rewards_ext = []
        rewards_int = []
        done_list = []
        prev_actions = []
        prev_rewards_ext = []
        prev_rewards_int = []
        actor_idx_list = []
        gamma_list = []
        beta_list = []
        for b in batchs:
            states.append(b["states"][:-1])
            n_states.append(b["states"][1:])
            actions.append(b["action"])
            next_invalid_actions_list.append(b["next_invalid_actions"])
            rewards_ext.append(b["reward_ext"])
            rewards_int.append(b["reward_int"])
            done_list.append(b["done"])
            prev_actions.append(b["prev_action"])
            prev_rewards_ext.append(b["prev_reward_ext"])
            prev_rewards_int.append(b["prev_reward_int"])
            actor_idx_list.append(b["actor"])
            gamma_list.append(self.gamma_list[b["actor"]])
            beta_list.append(self.beta_list[b["actor"]])
        states = np.asarray(states)
        n_states = np.asarray(n_states)
        rewards_ext = np.asarray(rewards_ext)
        rewards_int = np.asarray(rewards_int)
        prev_rewards_ext = np.asarray(prev_rewards_ext)
        prev_rewards_int = np.asarray(prev_rewards_int)
        beta_list = np.asarray(beta_list)

        actions_onehot = tf.one_hot(actions, self.config.action_num)
        prev_actions_onehot = tf.one_hot(prev_actions, self.config.action_num)
        actor_idx_onehot = tf.one_hot(actor_idx_list, self.config.actor_num)

        # ----------------------------------------
        # Q network
        # ----------------------------------------
        _params = [
            states,
            n_states,
            actions_onehot,
            next_invalid_actions_list,
            rewards_ext.reshape((-1, 1)),
            rewards_int.reshape((-1, 1)),
            done_list,
            prev_actions_onehot,
            prev_rewards_ext.reshape((-1, 1)),
            prev_rewards_int.reshape((-1, 1)),
            actor_idx_onehot,
            gamma_list,
            weights,
        ]
        td_error_ext, loss_ext = self._update_q(
            self.parameter.q_ext_online,
            self.parameter.q_ext_target,
            self.q_ext_optimizer,
            rewards_ext,
            *_params,
        )
        _info = {"loss_ext": loss_ext}

        if self.config.enable_intrinsic_reward:
            td_error_int, loss_int = self._update_q(
                self.parameter.q_int_online,
                self.parameter.q_int_target,
                self.q_int_optimizer,
                rewards_int,
                *_params,
            )

            # ----------------------------------------
            # embedding network
            # ----------------------------------------
            with tf.GradientTape() as tape:
                actions_probs = self.parameter.emb_network(states, n_states)
                emb_loss = self.emb_loss(actions_probs, actions_onehot)

            grads = tape.gradient(emb_loss, self.parameter.emb_network.trainable_variables)
            self.emb_optimizer.apply_gradients(zip(grads, self.parameter.emb_network.trainable_variables))

            # ----------------------------------------
            # lifelong network
            # ----------------------------------------
            lifelong_target_val = self.parameter.lifelong_target(states)
            with tf.GradientTape() as tape:
                lifelong_train_val = self.parameter.lifelong_train(states)
                lifelong_loss = self.lifelong_loss(lifelong_target_val, lifelong_train_val)

            grads = tape.gradient(lifelong_loss, self.parameter.lifelong_train.trainable_variables)
            self.lifelong_optimizer.apply_gradients(zip(grads, self.parameter.lifelong_train.trainable_variables))

            _info["loss_int"] = loss_int
            _info["emb_loss"] = emb_loss.numpy()
            _info["lifelong_loss"] = lifelong_loss.numpy()
        else:
            td_error_int = 0.0

        if self.config.disable_int_priority:
            td_errors = td_error_ext
        else:
            td_errors = td_error_ext + beta_list * td_error_int

        return td_errors, _info

    def _update_q(
        self,
        model_q_online,
        model_q_target,
        optimizer,
        rewards,
        #
        states,
        n_states,
        actions_onehot,
        next_invalid_actions_list,
        rewards_ext,
        rewards_int,
        done_list,
        prev_actions_onehot,
        prev_rewards_ext,
        prev_rewards_int,
        actor_idx_onehot,
        gamma_list,
        weights,
    ):
        target_q = self.parameter.calc_target_q(
            model_q_online,
            model_q_target,
            rewards,
            #
            n_states,
            actions_onehot,
            next_invalid_actions_list,
            rewards_ext,
            rewards_int,
            done_list,
            actor_idx_onehot,
            gamma_list,
        )

        with tf.GradientTape() as tape:
            q = model_q_online(states, prev_rewards_ext, prev_rewards_int, prev_actions_onehot, actor_idx_onehot)
            q = tf.reduce_sum(q * actions_onehot, axis=1)
            loss = self.q_loss(target_q * weights, q * weights)

        grads = tape.gradient(loss, model_q_online.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_q_online.trainable_variables))

        td_error = (target_q - q).numpy()
        loss = loss.numpy()

        return td_error, loss


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        self.dummy_state = np.full(self.config.observation_shape, self.config.dummy_state_val, dtype=np.float32)

        # actor
        self.beta_list = create_beta_list(self.config.actor_num)
        self.epsilon_list = create_epsilon_list(self.config.actor_num)
        self.gamma_list = create_gamma_list(self.config.actor_num)

        # ucb
        self.actor_index = -1
        self.ucb_recent = []
        self.ucb_actors_count = [1 for _ in range(self.config.actor_num)]  # 1回は保証
        self.ucb_actors_reward = [0.0 for _ in range(self.config.actor_num)]

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> None:
        self.recent_states = [self.dummy_state for _ in range(self.config.window_length + 1)]
        self.recent_states.pop(0)
        self.recent_states.append(state)
        self.invalid_actions = invalid_actions

        if self.training:
            # エピソード毎に actor を決める
            self.actor_index = self._calc_actor_index()
            self.beta = self.beta_list[self.actor_index]
            self.epsilon = self.epsilon_list[self.actor_index]
            self.gamma = self.gamma_list[self.actor_index]

        else:
            self.actor_index = 0
            self.epsilon = self.config.test_epsilon
            self.beta = self.config.test_beta

        self.prev_action = random.randint(0, self.config.action_num - 1)
        self.prev_reward_ext = 0
        self.prev_reward_int = 0

        # Q値取得用
        self.onehot_actor_idx = tf.one_hot(np.array(self.actor_index), self.config.actor_num)[np.newaxis, ...]

        # sliding-window UCB 用に報酬を保存
        self.episode_reward = 0.0

        # エピソードメモリ(エピソード毎に初期化)
        self.episodic_memory = collections.deque(maxlen=self.config.episodic_memory_capacity)

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
        return random.choice(np.where(ucbs == np.max(ucbs))[0])

    def call_policy(self, _state: np.ndarray, invalid_actions: List[int]) -> int:
        q, q_ext, _ = self._get_qval()

        if random.random() < self.epsilon:
            self.action = random.choice([a for a in range(self.config.action_num) if a not in invalid_actions])
        else:
            # valid actions以外は -inf にする
            q = [(-np.inf if a in invalid_actions else v) for a, v in enumerate(q)]

            # 最大値を選ぶ（複数はほぼないので無視）
            self.action = int(np.argmax(q))

        self.q_ext = q_ext[self.action]
        self.q = q[self.action]
        return self.action

    def _get_qval(self):
        prev_onehot_action = tf.one_hot(np.array(self.prev_action), self.config.action_num)[np.newaxis, ...]
        in_ = [
            np.asarray([self.recent_states[1:]]),
            np.array([[self.prev_reward_ext]]),
            np.array([[self.prev_reward_int]]),
            prev_onehot_action,
            self.onehot_actor_idx,
        ]
        q_ext = self.parameter.q_ext_online(*in_)[0].numpy()
        q_int = self.parameter.q_int_online(*in_)[0].numpy()
        q = q_ext + self.beta * q_int

        return q, q_ext, q_int

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward_ext: float,
        done: bool,
        next_invalid_actions: List[int],
    ):
        prev_action = self.prev_action
        prev_reward_ext = self.prev_reward_ext
        prev_reward_int = self.prev_reward_int

        self.episode_reward += reward_ext

        self.recent_states.pop(0)
        self.recent_states.append(next_state)
        self.invalid_actions = next_invalid_actions

        # 内部報酬
        if self.config.enable_intrinsic_reward:
            n_s = np.asarray([self.recent_states[1:]])
            episodic_reward = self._calc_episodic_reward(n_s)
            lifelong_reward = self._calc_lifelong_reward(n_s)
            reward_int = episodic_reward * lifelong_reward

            _info = {
                "episodic": episodic_reward,
                "lifelong": lifelong_reward,
                "reward_int": reward_int,
            }
        else:
            reward_int = 0.0
            _info = {}

        self.prev_action = self.action
        self.prev_reward_ext = reward_ext
        self.prev_reward_int = reward_int

        if not self.training:
            return _info

        batch = {
            "states": self.recent_states[:],
            "action": self.action,
            "next_invalid_actions": next_invalid_actions,
            "reward_ext": reward_ext,
            "reward_int": reward_int,
            "done": done,
            "prev_action": prev_action,
            "prev_reward_ext": prev_reward_ext,
            "prev_reward_int": prev_reward_int,
            "actor": self.actor_index,
        }

        if self.config.memory_name == "ReplayMemory":
            td_error = None
        elif not self.distributed:
            td_error = None
        else:
            _params = [
                n_s,
                tf.one_hot([self.action], self.config.action_num),
                [next_invalid_actions],
                np.array([prev_reward_ext]),
                np.array([prev_reward_int]),
                [done],
                self.onehot_actor_idx,
                [self.gamma],
            ]
            target_q_ext = self.parameter.calc_target_q(
                self.parameter.q_ext_online,
                self.parameter.q_ext_target,
                [reward_ext],
                *_params,
            )

            if self.config.disable_int_priority or not self.config.enable_intrinsic_reward:
                td_error = target_q_ext - self.q_ext
            else:
                target_q_int = self.parameter.calc_target_q(
                    self.parameter.q_int_online,
                    self.parameter.q_int_target,
                    [reward_int],
                    *_params,
                )
                td_error = (target_q_ext + self.beta * target_q_int) - self.q
            td_error = td_error[0]

        self.remote_memory.add(batch, td_error)

        return _info

    def _calc_episodic_reward(self, state):
        k = self.config.episodic_count_max
        epsilon = self.config.episodic_epsilon
        cluster_distance = self.config.episodic_cluster_distance
        c = self.config.episodic_pseudo_counts

        # 埋め込み関数から制御可能状態を取得
        cont_state = self.parameter.emb_network.predict(state)[0].numpy()

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
        rnd_target_val = self.parameter.lifelong_target(state)[0]
        rnd_train_val = self.parameter.lifelong_train(state)[0]

        # MSE
        error = np.square(rnd_target_val - rnd_train_val).mean()
        reward = 1 + error

        if reward < 1:
            reward = 1
        if reward > self.config.lifelong_max:
            reward = self.config.lifelong_max

        return reward

    def call_render(self, env: EnvRun) -> None:
        q, q_ext, q_int = self._get_qval()
        if self.config.enable_rescale:
            q = inverse_rescaling(q)
            q_ext = inverse_rescaling(q_ext)
            q_int = inverse_rescaling(q_int)
        invalid_actions = self.get_invalid_actions(env)

        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:6.3f} = {q_ext[a]:6.3f} + {self.beta:.3f} * {q_int[a]:6.3f}"

        render_discrete_action(invalid_actions, maxa, env, _render_sub)
