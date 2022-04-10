import collections
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from srl.base.rl import DiscreteActionConfig, RLParameter, RLRemoteMemory, RLTrainer, RLWorker
from srl.rl.functions.common import (
    calc_epsilon_greedy_probs,
    create_beta_list,
    create_epsilon_list,
    create_gamma_list,
    inverse_rescaling,
    random_choice_by_probs,
    rescaling,
)
from srl.rl.functions.dueling_network import create_dueling_network_layers
from srl.rl.functions.model import (
    ImageLayerType,
    create_input_layers,
    create_input_layers_lstm_stateful,
    create_input_layers_one_sequence,
)
from srl.rl.memory import factory
from srl.rl.registory import register
from tensorflow.keras import layers as kl

"""
DQN
    window_length               : x
    Target Network              : o
    Huber loss function         : o
    Delay update Target Network : o
    Experience Replay  : o
    Frame skip         : -
    Annealing e-greedy : o (option)
    Reward clip        : x
    Image preprocessor : -
Rainbow
    Double DQN               : o (config selection)
    Priority Experience Reply: o (config selection)
    Dueling Network          : o (config selection)
    Multi-Step learning      : (retrace)
    Noisy Network            : x
    Categorical DQN          : x
Recurrent Replay Distributed DQN(R2D2)
    LSTM                     : o
    Value function rescaling : o
Never Give Up(NGU)
    Intrinsic Reward : o
    UVFA             : TODO
    Retrace          : o (config selection)
Agent57
    Meta controller(sliding-window UCB) : o
    Intrinsic Reward split              : o
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
    dense_units: int = 512
    lstm_units: int = 512
    image_layer_type: ImageLayerType = ImageLayerType.DQN
    burnin: int = 40
    q_ext_lr: float = 0.001
    q_int_lr: float = 0.001
    batch_size: int = 32
    target_model_update_interval: int = 100

    # retrace
    multisteps: int = 1
    retrace_h: float = 1.0

    # double dqn
    enable_double_dqn: bool = True

    # DuelingNetwork
    enable_dueling_network: bool = True
    dueling_network_type: str = "average"
    dueling_dense_units: int = 512

    # Priority Experience Replay
    capacity: int = 100_000
    memory_name: str = "RankBaseMemory"
    memory_warmup_size: int = 1000
    memory_alpha: float = 0.6
    memory_beta_initial: float = 0.4
    memory_beta_steps: int = 1_000_000

    # ucb(160,0.5 or 3600,0.01)
    actor_num: int = 32
    ucb_window_size: int = 160  # UCB上限
    ucb_epsilon: float = 0.5  # UCBを使う確率
    ucb_beta: float = 1  # UCBのβ

    # episodic
    episodic_lr: float = 0.0005
    episodic_count_max: int = 10  # k
    episodic_epsilon: float = 0.001
    episodic_cluster_distance: float = 0.008
    episodic_memory_capacity: int = 30000
    episodic_pseudo_counts: float = 0.1  # 疑似カウント定数

    # lifelong
    lifelong_lr: float = 0.00001
    lifelong_max: float = 5.0  # L

    dummy_state_val: float = 0.0

    @staticmethod
    def getName() -> str:
        return "Agent57"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.burnin >= 0
        assert self.multisteps >= 1
        assert self.memory_warmup_size < self.capacity
        assert self.batch_size < self.memory_warmup_size


register(Config, __name__)


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()
        # input_shape: (batch_size, input_sequence(timestamps), observation_shape)

        # timestamps=1(stateful)
        in_state, c = create_input_layers_lstm_stateful(
            config.batch_size,
            1,
            config.env_observation_shape,
            config.env_observation_type,
            config.image_layer_type,
        )

        # lstm
        c = kl.LSTM(config.lstm_units, stateful=True, name="lstm")(c)

        if config.enable_dueling_network:
            c = create_dueling_network_layers(
                c, config.nb_actions, config.dueling_dense_units, config.dueling_network_type
            )
        else:
            c = kl.Dense(config.dense_units, activation="relu", kernel_initializer="he_normal")(c)
            c = kl.Dense(config.nb_actions, kernel_initializer="truncated_normal")(c)

        self.model = keras.Model(in_state, c)
        self.lstm_layer = self.model.get_layer("lstm")

        # 重みを初期化
        in_shape = (1,) + config.env_observation_shape
        dummy_state = np.zeros(shape=(config.batch_size,) + in_shape, dtype=np.float32)
        val, states = self(dummy_state, None)
        assert val.shape == (config.batch_size, config.nb_actions)

    def call(self, state, hidden_states):
        self.lstm_layer.reset_states(hidden_states)
        return self.model(state), self.get_hidden_state()

    def get_hidden_state(self):
        return [self.lstm_layer.states[0].numpy(), self.lstm_layer.states[1].numpy()]

    def init_hidden_state(self):
        self.lstm_layer.reset_states()
        return self.get_hidden_state()


# ------------------------------------------------------
# エピソード記憶部(episodic_reward)
# ------------------------------------------------------
class _EmbeddingNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        in_state, c = create_input_layers_one_sequence(
            config.env_observation_shape,
            config.env_observation_type,
            config.image_layer_type,
        )

        c = kl.Dense(
            32,
            activation="relu",
            kernel_initializer="he_normal",
            bias_initializer=keras.initializers.constant(0.001),
        )(c)
        self.model = keras.Model(in_state, c)

        # out layer
        self.concatrate = kl.Concatenate()
        self.d1 = kl.Dense(128, activation="relu", kernel_initializer="he_normal")
        c = kl.LayerNormalization()(c)
        self.out = kl.Dense(config.nb_actions, activation="softmax")

        # 重みを初期化
        dummy_state = np.zeros(shape=(1,) + config.env_observation_shape, dtype=np.float32)
        val = self(dummy_state, dummy_state)
        assert val.shape == (1, config.nb_actions)

    def call(self, state1, state2):
        c1 = self.model(state1)
        c2 = self.model(state2)
        c = self.concatrate([c1, c2])
        c = self.d1(c)
        c = self.out(c)
        return c

    def predict(self, state):
        return self.model(state)


# ------------------------------------------------------
# 生涯記憶部(life long novelty module)
# ------------------------------------------------------
class _LifelongNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        in_state, c = create_input_layers_one_sequence(
            config.env_observation_shape,
            config.env_observation_type,
            config.image_layer_type,
        )

        c = kl.Dense(
            128,
            activation="relu",
            kernel_initializer="he_normal",
            bias_initializer="he_normal",
        )(c)
        c = kl.LayerNormalization()(c)
        self.model = keras.Model(in_state, c)

        # 重みを初期化
        dummy_state = np.zeros(shape=(1,) + config.env_observation_shape, dtype=np.float32)
        val = self(dummy_state)
        assert val.shape == (1, 128)

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
        self.emb_network.summary()
        self.lifelong_target.model.summary()


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
        self.invalid_memory = deque(maxlen=self.config.capacity)

    def length(self) -> int:
        return len(self.memory)

    def restore(self, data: Any) -> None:
        self.memory.restore(data[0])
        self.invalid_memory = data[1]

    def backup(self):
        d = [self.memory.backup(), self.invalid_memory]
        return d

    # ---------------------------

    def add(self, batch, priority):
        self.memory.add(batch, priority)

    def sample(self, step: int) -> Tuple[list, list, list]:
        return self.memory.sample(self.config.batch_size, step)

    def update(self, indexes: List[int], batchs: List[Any], priorities: List[float]) -> None:
        self.memory.update(indexes, batchs, priorities)

    def length_invalid(self) -> int:
        return len(self.invalid_memory)

    def add_invalid(self, batch):
        self.invalid_memory.append(batch)

    def sample_invalid(self):
        return random.sample(self.invalid_memory, self.config.batch_size)

    def clear_invalid(self):
        return self.invalid_memory.clear()


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.memory = cast(RemoteMemory, self.memory)

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

    def get_train_count(self):
        return self.train_count

    def train(self):

        if self.memory.length() < self.config.memory_warmup_size:
            return {}

        indexes, batchs, weights = self.memory.sample(self.train_count)
        td_errors, info = self._train_on_batchs(batchs, weights)
        priorities = np.abs(td_errors) + 0.0001
        self.memory.update(indexes, batchs, priorities)

        # invalid action TODO
        # mem_invalid_len = self.memory.length_invalid()
        # if mem_invalid_len > self.config.memory_warmup_size:
        #    batchs = self.memory.sample_invalid()
        #    self._train_on_batchs(batchs, [1 for _ in range(self.config.batch_size)])

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_ext_target.set_weights(self.parameter.q_ext_online.get_weights())
            self.parameter.q_int_target.set_weights(self.parameter.q_int_online.get_weights())

        self.train_count += 1
        return info

    def _train_on_batchs(self, batchs, weights):

        # burnin=2
        # multisteps=3
        # states  [0,1,2,3,4,5,6]
        # burnin   o o
        # state        o,o
        # n_state1       o,o
        # n_state2         o,o
        # n_state3           o,o

        # (batch, dict[x], multisteps) -> (multisteps, batch, x)
        states_list = []
        for i in range(self.config.multisteps + 1):
            states_list.append(np.asarray([[b["states"][i]] for b in batchs]))
        actions_list = []
        mu_probs_list = []
        rewards_list_ext = []
        rewards_list_int = []
        dones_list = []
        for i in range(self.config.multisteps):
            actions_list.append([b["actions"][i] for b in batchs])
            rewards_list_ext.append([b["rewards_ext"][i] for b in batchs])
            rewards_list_int.append([b["rewards_int"][i] for b in batchs])
            mu_probs_list.append([b["probs"][i] for b in batchs])
            dones_list.append([b["dones"][i] for b in batchs])

        one_states = []
        one_n_states = []
        one_actions = []
        gamma_list = []
        beta_list = []
        for b in batchs:
            one_states.append(b["states"][0])
            one_n_states.append(b["states"][1])
            one_actions.append(b["actions"][0])
            gamma_list.append(self.gamma_list[b["actor"]])
            beta_list.append(self.beta_list[b["actor"]])
        one_states = np.asarray(one_states)
        one_n_states = np.asarray(one_n_states)
        one_actions_onehot = tf.one_hot(one_actions, self.config.nb_actions)

        # hidden_states
        states_h_ext = []
        states_c_ext = []
        states_h_int = []
        states_c_int = []
        for b in batchs:
            states_h_ext.append(b["hidden_states_ext"][0])
            states_c_ext.append(b["hidden_states_ext"][1])
            states_h_int.append(b["hidden_states_int"][0])
            states_c_int.append(b["hidden_states_int"][1])
        hidden_states_ext = [np.asarray(states_h_ext), np.asarray(states_c_ext)]
        hidden_states_int = [np.asarray(states_h_int), np.asarray(states_c_int)]

        # burnin
        for i in range(self.config.burnin):
            burnin_state = np.asarray([[b["burnin_states"][i]] for b in batchs])
            _, hidden_states_ext = self.parameter.q_ext_online(burnin_state, hidden_states_ext)
            _, hidden_states_int = self.parameter.q_int_online(burnin_state, hidden_states_int)

        _params = [
            states_list,
            actions_list,
            mu_probs_list,
            dones_list,
            weights,
            gamma_list,
            0,
        ]
        _, _, td_error_ext, _, loss_ext = self._train_steps(
            self.parameter.q_ext_online,
            self.parameter.q_ext_target,
            self.q_ext_optimizer,
            rewards_list_ext,
            hidden_states_ext,
            *_params,
        )
        _, _, td_error_int, _, loss_int = self._train_steps(
            self.parameter.q_int_online,
            self.parameter.q_int_target,
            self.q_ext_optimizer,
            rewards_list_int,
            hidden_states_int,
            *_params,
        )
        td_errors = td_error_ext
        # td_errors = td_error_ext + beta_list * td_error_int

        # ----------------------------------------
        # embedding network
        # ----------------------------------------
        with tf.GradientTape() as tape:
            actions_prebs = self.parameter.emb_network(one_states, one_n_states)
            emb_loss = self.emb_loss(actions_prebs, one_actions_onehot)

        grads = tape.gradient(emb_loss, self.parameter.emb_network.trainable_variables)
        self.emb_optimizer.apply_gradients(zip(grads, self.parameter.emb_network.trainable_variables))

        # ----------------------------------------
        # lifelong network
        # ----------------------------------------
        lifelong_target_val = self.parameter.lifelong_target(one_states)
        with tf.GradientTape() as tape:
            lifelong_train_val = self.parameter.lifelong_train(one_states)
            lifelong_loss = self.lifelong_loss(lifelong_target_val, lifelong_train_val)

        grads = tape.gradient(lifelong_loss, self.parameter.lifelong_train.trainable_variables)
        self.lifelong_optimizer.apply_gradients(zip(grads, self.parameter.lifelong_train.trainable_variables))

        return td_errors, {
            "loss_ext": loss_ext,
            "loss_int": loss_int,
            "emb_loss": emb_loss.numpy(),
            "lifelong_loss": lifelong_loss.numpy(),
        }

    # Q値(LSTM hidden states)の予測はforward、td_error,retraceはbackで予測する必要あり
    def _train_steps(
        self,
        model_q_online,
        model_q_target,
        optimizer,
        rewards_list,
        hidden_states,
        #
        states_list,
        actions_list,
        mu_probs_list,
        dones_list,
        weights,
        gamma_list,
        idx,
    ):

        # 最後
        if idx == self.config.multisteps:
            n_states = states_list[idx]
            n_q_target, _ = model_q_target(n_states, hidden_states)
            n_q, _ = model_q_online(n_states, hidden_states)
            n_q_target = tf.stop_gradient(n_q_target).numpy()
            n_q = tf.stop_gradient(n_q).numpy()
            return n_q, n_q_target, np.zeros(self.config.batch_size), 1.0, 0

        states = states_list[idx]
        n_states = states_list[idx + 1]
        actions = actions_list[idx]
        dones = dones_list[idx]
        rewards = rewards_list[idx]
        mu_probs = mu_probs_list[idx]

        q_target, _ = model_q_target(states, hidden_states)
        q_target = tf.stop_gradient(q_target).numpy()
        with tf.GradientTape() as tape:
            q, n_hidden_states = model_q_online(states, hidden_states)

            n_q, n_q_target, n_td_error, retrace, _ = self._train_steps(
                model_q_online,
                model_q_target,
                optimizer,
                rewards_list,
                n_hidden_states,
                states_list,
                actions_list,
                mu_probs_list,
                dones_list,
                weights,
                gamma_list,
                idx + 1,
            )
            target_q = []
            for i in range(self.config.batch_size):
                if dones[i]:
                    gain = rewards[i]
                else:
                    # DoubleDQN: indexはonlineQから選び、値はtargetQを選ぶ
                    if self.config.enable_double_dqn:
                        n_act_idx = np.argmax(n_q[i])
                    else:
                        n_act_idx = np.argmax(n_q_target[i])
                    maxq = n_q_target[i][n_act_idx]
                    maxq = inverse_rescaling(maxq)
                    gain = rewards[i] + gamma_list[i] * maxq
                gain = rescaling(gain)
                target_q.append(gain)
            target_q = np.asarray(target_q).astype(np.float32)

            # retrace
            _retrace = []
            for i in range(self.config.batch_size):
                pi_probs = calc_epsilon_greedy_probs(
                    n_q[i],
                    [a for a in range(self.config.nb_actions)],
                    0.0,
                    self.config.nb_actions,
                )
                r = self.config.retrace_h * np.minimum(1, pi_probs[actions[i]] / mu_probs[i])
                _retrace.append(r)
            retrace *= np.asarray(_retrace)

            target_q += gamma_list * retrace * n_td_error

            # 現在選んだアクションのQ値
            action_onehot = tf.one_hot(actions, self.config.nb_actions)
            q_onehot = tf.reduce_sum(q * action_onehot, axis=1)

            loss = self.q_loss(target_q * weights, q_onehot * weights)

        grads = tape.gradient(loss, model_q_online.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_q_online.trainable_variables))

        n_td_error = (target_q - q_onehot).numpy() + gamma_list * retrace * n_td_error
        q = tf.stop_gradient(q).numpy()
        return q, q_target, n_td_error, retrace, loss.numpy()


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.memory = cast(RemoteMemory, self.memory)

        self.dummy_state = np.full(self.config.env_observation_shape, self.config.dummy_state_val, dtype=np.float32)
        self.invalid_action_reward = -1

        # actor
        self.beta_list = create_beta_list(self.config.actor_num)
        self.epsilon_list = create_epsilon_list(self.config.actor_num)
        self.gamma_list = create_gamma_list(self.config.actor_num)

        # ucb
        self.actor_index = -1
        self.ucb_recent = []
        self.ucb_actors_count = [1 for _ in range(self.config.actor_num)]  # 1回は保証
        self.ucb_actors_reward = [0.0 for _ in range(self.config.actor_num)]

    def on_reset(self, state: np.ndarray, valid_actions: List[int], _) -> None:
        self.recent_states = [self.dummy_state for _ in range(self.config.burnin + self.config.multisteps + 1)]
        self.recent_actions = [random.randint(0, self.config.nb_actions - 1) for _ in range(self.config.multisteps)]
        self.recent_probs = [1.0 / self.config.nb_actions for _ in range(self.config.multisteps)]
        self.recent_rewards_ext = [0.0 for _ in range(self.config.multisteps)]
        self.recent_rewards_int = [0.0 for _ in range(self.config.multisteps)]
        self.recent_done = [False for _ in range(self.config.multisteps)]
        self.recent_valid_actions = [[] for _ in range(self.config.multisteps + 1)]

        self.hidden_state_ext = self.parameter.q_ext_online.init_hidden_state()
        self.hidden_state_int = self.parameter.q_int_online.init_hidden_state()
        self.recent_hidden_states_ext = [
            [self.hidden_state_ext[0][0], self.hidden_state_ext[1][0]]
            for _ in range(self.config.burnin + self.config.multisteps + 1)
        ]
        self.recent_hidden_states_int = [
            [self.hidden_state_int[0][0], self.hidden_state_int[1][0]]
            for _ in range(self.config.burnin + self.config.multisteps + 1)
        ]

        self.recent_states.pop(0)
        self.recent_states.append(state.astype(np.float32))
        self.recent_valid_actions.pop(0)
        self.recent_valid_actions.append(valid_actions)

        if self.training:
            # エピソード毎に actor を決める
            self.actor_index = self._calc_actor_index()
            self.beta = self.beta_list[self.actor_index]
            self.epsilon = self.epsilon_list[self.actor_index]
            self.gamma = self.gamma_list[self.actor_index]

        else:
            self.epsilon = self.config.test_epsilon
            self.beta = self.config.test_beta

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

    def policy(self, state: np.ndarray, valid_actions: List[int], _) -> Tuple[int, Any]:
        state = np.asarray([[state]] * self.config.batch_size)

        q_ext, self.hidden_state_ext = self.parameter.q_ext_online(state, self.hidden_state_ext)
        q_int, self.hidden_state_int = self.parameter.q_int_online(state, self.hidden_state_int)
        q = q_ext[0] + self.beta * q_int[0]
        q = q.numpy()

        probs = calc_epsilon_greedy_probs(q, valid_actions, self.epsilon, self.config.nb_actions)
        action = random_choice_by_probs(probs)

        return action, (action, probs[action], q[action])

    def on_step(
        self,
        state: np.ndarray,
        action_: Any,
        next_state: np.ndarray,
        reward_ext: float,
        done: bool,
        valid_actions: List[int],
        next_valid_actions: List[int],
        _,
    ):
        self.episode_reward += reward_ext

        # 内部報酬
        n_s = np.asarray([next_state])
        episodic_reward = self._calc_episodic_reward(n_s)
        lifelong_reward = self._calc_lifelong_reward(n_s)
        reward_int = episodic_reward * lifelong_reward

        _info = {
            "episodic": episodic_reward,
            "lifelong": lifelong_reward,
            "reward_int": reward_int,
        }
        if not self.training:
            return _info

        action = action_[0]
        prob = action_[1]
        q = action_[2]

        # invalid_action_reward
        if self.invalid_action_reward > reward_ext - 1:
            self.invalid_action_reward = reward_ext - 1
            self.memory.clear_invalid()

        self.recent_states.pop(0)
        self.recent_states.append(next_state.astype(np.float32))
        self.recent_actions.pop(0)
        self.recent_actions.append(action)
        self.recent_probs.pop(0)
        self.recent_probs.append(prob)
        self.recent_rewards_ext.pop(0)
        self.recent_rewards_ext.append(reward_ext)
        self.recent_rewards_int.pop(0)
        self.recent_rewards_int.append(reward_int)
        self.recent_done.pop(0)
        self.recent_done.append(done)
        self.recent_valid_actions.pop(0)
        self.recent_valid_actions.append(next_valid_actions)
        self.recent_hidden_states_ext.pop(0)
        self.recent_hidden_states_ext.append(
            [
                self.hidden_state_ext[0][0],
                self.hidden_state_ext[1][0],
            ]
        )
        self.recent_hidden_states_int.pop(0)
        self.recent_hidden_states_int.append(
            [
                self.hidden_state_int[0][0],
                self.hidden_state_int[1][0],
            ]
        )

        priority = self._add_memory(q, None)

        if done:
            # 残りstepも追加
            for _ in range(len(self.recent_rewards_ext) - 1):
                self.recent_states.pop(0)
                self.recent_states.append(self.dummy_state)
                self.recent_actions.pop(0)
                self.recent_actions.append(random.randint(0, self.config.nb_actions - 1))
                self.recent_probs.pop(0)
                self.recent_probs.append(1.0 / self.config.nb_actions)
                self.recent_rewards_ext.pop(0)
                self.recent_rewards_ext.append(0.0)
                self.recent_rewards_int.pop(0)
                self.recent_rewards_int.append(0.0)
                self.recent_done.pop(0)
                self.recent_done.append(True)
                self.recent_valid_actions.pop(0)
                self.recent_valid_actions.append([])
                self.recent_hidden_states_ext.pop(0)
                self.recent_hidden_states_int.pop(0)

                self._add_memory(q, priority)

        # --- valid action
        if False:  # TODO
            states = self.recent_states[:]
            states[-1] = self.dummy_state
            bundle_states = self.recent_bundle_states[:]
            bundle_states[-1] = states
            rewards = self.recent_rewards[:]
            rewards[-1] = self.invalid_action_reward
            probs = self.recent_probs[:]
            probs[-1] = 1.0
            dones = self.recent_done[:]
            dones[-1] = True
            for a in range(self.config.nb_actions):
                if a in valid_actions:
                    continue
                actions = self.recent_actions[:]
                actions[-1] = a

                batch = {
                    "states": bundle_states,
                    "actions": actions,
                    "probs": probs,
                    "rewards": rewards,
                    "dones": dones,
                    "valid_actions": self.recent_valid_actions[:],
                }
                self.memory.add_invalid(batch)

        return _info

    def _add_memory(self, q, priority):

        batch = {
            "states": self.recent_states[self.config.burnin :],
            "actions": self.recent_actions[:],
            "probs": self.recent_probs[:],
            "rewards_ext": self.recent_rewards_ext[:],
            "rewards_int": self.recent_rewards_int[:],
            "dones": self.recent_done[:],
            "actor": self.actor_index,
            "valid_actions": self.recent_valid_actions[:],
            "burnin_states": self.recent_states[: self.config.burnin],
            "hidden_states_ext": self.recent_hidden_states_ext[0],
            "hidden_states_int": self.recent_hidden_states_int[0],
        }

        # priority
        if priority is None:
            if self.config.memory_name == "ReplayMemory":
                priority = 1
            else:
                # TODO
                priority = 1
                # target_q = self.parameter.calc_target_q([batch])[0]
                # priority = abs(target_q - q) + 0.0001

        self.memory.add(batch, priority)
        return priority

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

    def render(
        self,
        state: np.ndarray,
        valid_actions: List[int],
        action_to_str,
    ) -> None:
        states = np.asarray([[state]] * self.config.batch_size)
        q_ext, _ = self.parameter.q_ext_online(states, self.hidden_state_ext)
        q_int, _ = self.parameter.q_int_online(states, self.hidden_state_int)
        q_ext = q_ext[0].numpy()
        q_int = q_int[0].numpy()
        q = q_ext + self.beta * q_ext

        n_s = np.asarray([state])
        episodic_reward = self._calc_episodic_reward(n_s)
        lifelong_reward = self._calc_lifelong_reward(n_s)
        reward_int = episodic_reward * lifelong_reward

        print("episodic_reward", episodic_reward)
        print("lifelong_reward", lifelong_reward)
        print("reward_int", reward_int)

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
            s += f"{action_to_str(a)}: {q[a]:5.3f}"
            s += f"{a:2d}: {q[a]:5.3f} = {q_ext[a]:5.3f} + {self.beta} * {q_int[a]:5.3f}"
            print(s)


if __name__ == "__main__":
    pass
