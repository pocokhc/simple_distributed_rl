import collections
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kl

from srl.base.define import EnvObservationType, RLObservationType
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.processor import Processor
from srl.base.rl.processors.image_processor import ImageProcessor
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import PriorityExperienceReplay
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
from srl.rl.models.tf.dqn_image_block import DQNImageBlock
from srl.rl.models.tf.dueling_network import DuelingNetworkBlock
from srl.rl.models.tf.input_layer import create_input_layer

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
class Config(DiscreteActionConfig):

    # test
    test_epsilon: float = 0
    test_beta: float = 0

    # model
    lstm_units: int = 512
    cnn_block: kl.Layer = DQNImageBlock
    cnn_block_kwargs: dict = None
    hidden_layer_sizes: Tuple[int, ...] = (512,)
    activation: str = "relu"

    q_ext_lr: float = 0.0001
    q_int_lr: float = 0.0001
    batch_size: int = 64
    target_model_update_interval: int = 1500

    # lstm
    burnin: int = 5
    sequence_length: int = 5

    # rescale
    enable_rescale: bool = False

    # retrace
    enable_retrace: bool = True
    retrace_h: float = 1.0

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
        if self.cnn_block_kwargs is None:
            self.cnn_block_kwargs = {}

    def set_processor(self) -> List[Processor]:
        return [
            ImageProcessor(
                image_type=EnvObservationType.GRAY_2ch,
                resize=(84, 84),
                enable_norm=True,
            )
        ]

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.CONTINUOUS

    @staticmethod
    def getName() -> str:
        return "Agent57"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.burnin >= 0
        assert self.sequence_length >= 1
        assert self.memory_warmup_size < self.capacity
        assert self.batch_size < self.memory_warmup_size
        assert len(self.hidden_layer_sizes) > 0
        assert len(self.episodic_hidden_layer_sizes1) > 0
        assert len(self.episodic_hidden_layer_sizes2) > 0
        assert len(self.lifelong_hidden_layer_sizes) > 0


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

        # --- in block
        in_state, c, use_image_head = create_input_layer(config.observation_shape, config.env_observation_type)
        if use_image_head:
            c = config.cnn_block(**config.cnn_block_kwargs)(c)
            c = kl.Flatten()(c)
        self.in_block = kl.TimeDistributed(keras.Model(in_state, c))

        # --- lstm
        self.lstm_layer = kl.LSTM(config.lstm_units, return_sequences=True, return_state=True)

        # --- out block
        in_state = c = kl.Input(config.lstm_units)

        for i in range(len(config.hidden_layer_sizes) - 1):
            c = kl.Dense(
                config.hidden_layer_sizes[i],
                activation=config.activation,
                kernel_initializer="he_normal",
            )(c)

        if config.enable_dueling_network:
            c = DuelingNetworkBlock(
                config.action_num,
                config.hidden_layer_sizes[-1],
                config.dueling_network_type,
                activation=config.activation,
            )(c)
        else:
            c = kl.Dense(config.hidden_layer_sizes[-1], activation=config.activation, kernel_initializer="he_normal")(
                c
            )
            c = kl.Dense(
                config.action_num, kernel_initializer="truncated_normal", bias_initializer="truncated_normal"
            )(c)

        self.out_block = kl.TimeDistributed(keras.Model(in_state, c))

        # 重みを初期化
        dummy1 = np.zeros(
            shape=(config.batch_size, config.sequence_length) + config.observation_shape, dtype=np.float32
        )
        dummy2 = np.zeros(shape=(config.batch_size, config.sequence_length, 1), dtype=np.float32)
        dummy3 = np.zeros(shape=(config.batch_size, config.sequence_length, 1), dtype=np.float32)
        dummy4 = np.zeros(shape=(config.batch_size, config.sequence_length, config.action_num), dtype=np.float32)
        dummy5 = np.zeros(shape=(config.batch_size, config.sequence_length, config.actor_num), dtype=np.float32)
        val, _ = self(dummy1, dummy2, dummy3, dummy4, dummy5, None)
        assert val.shape == (config.batch_size, config.sequence_length, config.action_num)

    def call(self, state, reward_ext, reward_int, onehot_action, onehot_actor, hidden_states, training=False):
        x = self.in_block(state, training=training)

        # UVFA
        uvfa_list = [x]
        if self.input_ext_reward:
            uvfa_list.append(reward_ext)
        if self.input_int_reward:
            uvfa_list.append(reward_int)
        if self.input_action:
            uvfa_list.append(onehot_action)
        uvfa_list.append(onehot_actor)
        x = tf.concat(uvfa_list, axis=2)

        x, h, c = self.lstm_layer(x, initial_state=hidden_states, training=training)
        x = self.out_block(x, training=training)
        return x, [h, c]

    def init_hidden_state(self):
        return self.lstm_layer.cell.get_initial_state(batch_size=1, dtype=tf.float32)


# ------------------------------------------------------
# エピソード記憶部(episodic_reward)
# ------------------------------------------------------
class _EmbeddingNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        # in model
        in_state, c, use_image_head = create_input_layer(
            config.observation_shape,
            config.env_observation_type,
        )
        if use_image_head:
            c = config.cnn_block(**config.cnn_block_kwargs)(c)
            c = kl.Flatten()(c)

        # hidden
        for h in config.episodic_hidden_layer_sizes1:
            c = kl.Dense(
                h,
                activation="relu",
                kernel_initializer="he_normal",
                bias_initializer=keras.initializers.constant(0.001),
            )(c)
        self.model1 = keras.Model(in_state, c, name="EmbeddingNetwork_predict")

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
        self.model2 = keras.Model([in1, in2], c, name="EmbeddingNetwork")

        # 重みを初期化
        dummy_state = np.zeros(shape=(1,) + config.observation_shape, dtype=np.float32)
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

        in_state, c, use_image_head = create_input_layer(
            config.observation_shape,
            config.env_observation_type,
        )
        if use_image_head:
            c = config.cnn_block(**config.cnn_block_kwargs)(c)
            c = kl.Flatten()(c)

        # hidden
        for h in config.lifelong_hidden_layer_sizes:
            c = kl.Dense(
                h,
                activation="relu",
                kernel_initializer="he_normal",
                bias_initializer="he_normal",
            )(c)
        c = kl.LayerNormalization()(c)
        self.model = keras.Model(in_state, c, name="LifelongNetwork")

        # 重みを初期化
        dummy_state = np.zeros(shape=(1,) + config.observation_shape, dtype=np.float32)
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

    def call_restore(self, data: Any, **kwargs) -> None:
        self.q_ext_online.set_weights(data[0])
        self.q_ext_target.set_weights(data[0])
        self.q_int_online.set_weights(data[1])
        self.q_int_target.set_weights(data[1])
        self.emb_network.set_weights(data[2])
        self.lifelong_target.set_weights(data[3])
        self.lifelong_train.set_weights(data[4])

    def call_backup(self, **kwargs):
        d = [
            self.q_ext_online.get_weights(),
            self.q_int_online.get_weights(),
            self.emb_network.get_weights(),
            self.lifelong_target.get_weights(),
            self.lifelong_train.get_weights(),
        ]
        return d

    def summary(self, **kwargs):
        self.q_ext_online.summary(**kwargs)
        self.emb_network.model1.summary(**kwargs)
        self.emb_network.model2.summary(**kwargs)
        self.lifelong_target.model.summary(**kwargs)


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
        self.discount_list = create_discount_list(self.config.actor_num)
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

        # (batch, dict[x], step) -> (batch, step, x)
        burnin_states = []
        burnin_rewards_ext = []
        burnin_rewards_int = []
        instep_states = []
        instep_rewards_ext = []
        instep_rewards_int = []
        actions = []
        step_probs = []
        step_dones = []
        step_invalid_actions_list = []
        for b in batchs:
            burnin_states.append([s for s in b["states"][: self.config.burnin]])
            burnin_rewards_ext.append([s for s in b["rewards_ext"][: self.config.burnin]])
            burnin_rewards_int.append([s for s in b["rewards_int"][: self.config.burnin]])
            instep_states.append([s for s in b["states"][self.config.burnin :]])
            instep_rewards_ext.append([s for s in b["rewards_ext"][self.config.burnin :]])
            instep_rewards_int.append([s for s in b["rewards_int"][self.config.burnin :]])
            actions.append([s for s in b["actions"]])
            step_probs.append([s for s in b["probs"]])
            step_dones.append([s for s in b["dones"]])
            step_invalid_actions_list.append([s for s in b["invalid_actions"]])
        burnin_states = np.asarray(burnin_states)
        burnin_rewards_ext = np.asarray(burnin_rewards_ext)
        burnin_rewards_int = np.asarray(burnin_rewards_int)
        instep_states = np.asarray(instep_states)
        instep_rewards_ext = np.asarray(instep_rewards_ext)
        instep_rewards_int = np.asarray(instep_rewards_int)

        actions_onehot = tf.one_hot(actions, self.config.action_num)
        burnin_actions_onehot = actions_onehot[:, : self.config.burnin, ...]
        instep_actions_onehot = actions_onehot[:, self.config.burnin :, ...]
        step_actions_onehot = instep_actions_onehot[:, 1:, ...]

        step_actions = np.array(actions)[:, self.config.burnin + 1 :]
        step_rewards_ext = instep_rewards_ext[:, 1:]
        step_rewards_int = instep_rewards_int[:, 1:]

        # other
        actor_list = []
        discount_list = []
        beta_list = []
        for b in batchs:
            actor_list.append([b["actor"]])
            discount_list.append(self.discount_list[b["actor"]])
            beta_list.append(self.beta_list[b["actor"]])
        discount_list = np.asarray(discount_list)
        beta_list = np.asarray(beta_list)

        # actor_onehot
        actor_onehot = tf.one_hot(actor_list, self.config.actor_num)
        burnin_actor_onehot = tf.tile(actor_onehot, [1, self.config.burnin, 1])
        instep_actor_onehot = tf.tile(actor_onehot, [1, self.config.sequence_length + 1, 1])

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
        hidden_states_ext = [tf.stack(states_h_ext), tf.stack(states_c_ext)]
        hidden_states_int = [tf.stack(states_h_int), tf.stack(states_c_int)]
        hidden_states_ext_t = hidden_states_ext
        hidden_states_int_t = hidden_states_int

        in_burnin = [
            burnin_states,
            burnin_rewards_ext[..., np.newaxis],
            burnin_rewards_int[..., np.newaxis],
            burnin_actions_onehot,
            burnin_actor_onehot,
        ]
        in_steps = [
            instep_states,
            instep_rewards_ext[..., np.newaxis],
            instep_rewards_int[..., np.newaxis],
            instep_actions_onehot,
            instep_actor_onehot,
        ]
        _params = [
            in_burnin,
            in_steps,
            step_actions,
            step_probs,
            step_dones,
            step_invalid_actions_list,
            step_actions_onehot,
            discount_list,
            np.asarray(weights).reshape(-1, 1),
        ]
        td_error_ext, loss_ext = self._train_q(
            self.parameter.q_ext_online,
            self.parameter.q_ext_target,
            self.q_ext_optimizer,
            step_rewards_ext,
            hidden_states_ext,
            hidden_states_ext_t,
            *_params,
        )
        _info = {"loss_ext": loss_ext}

        if self.config.enable_intrinsic_reward:
            td_error_int, loss_int = self._train_q(
                self.parameter.q_int_online,
                self.parameter.q_int_target,
                self.q_int_optimizer,
                step_rewards_int,
                hidden_states_int,
                hidden_states_int_t,
                *_params,
            )
            _info["loss_int"] = loss_int

            # embedding lifelong (batch, seq_len, x) -> (batch, x)
            one_states = instep_states[:, 1, ...]
            one_n_states = instep_states[:, 2, ...]
            one_actions_onehot = instep_actions_onehot[:, 1, :]

            # ----------------------------------------
            # embedding network
            # ----------------------------------------
            with tf.GradientTape() as tape:
                actions_probs = self.parameter.emb_network(one_states, one_n_states)
                emb_loss = self.emb_loss(actions_probs, one_actions_onehot)
                emb_loss += tf.reduce_sum(self.parameter.emb_network.losses)

            grads = tape.gradient(emb_loss, self.parameter.emb_network.trainable_variables)
            self.emb_optimizer.apply_gradients(zip(grads, self.parameter.emb_network.trainable_variables))

            # ----------------------------------------
            # lifelong network
            # ----------------------------------------
            lifelong_target_val = self.parameter.lifelong_target(one_states)
            with tf.GradientTape() as tape:
                lifelong_train_val = self.parameter.lifelong_train(one_states)
                lifelong_loss = self.lifelong_loss(lifelong_target_val, lifelong_train_val)
                lifelong_loss += tf.reduce_sum(self.parameter.lifelong_train.losses)

            grads = tape.gradient(lifelong_loss, self.parameter.lifelong_train.trainable_variables)
            self.lifelong_optimizer.apply_gradients(zip(grads, self.parameter.lifelong_train.trainable_variables))

            _info["emb_loss"] = emb_loss.numpy()
            _info["lifelong_loss"] = lifelong_loss.numpy()

        else:
            td_error_int = 0

        if self.config.disable_int_priority:
            td_errors = td_error_ext
        else:
            td_errors = td_error_ext + beta_list * td_error_int

        return td_errors, _info

    def _train_q(
        self,
        model_q_online,
        model_q_target,
        optimizer,
        step_rewards,
        hidden_states,
        hidden_states_t,
        #
        in_burnin,
        in_steps,
        step_actions,
        step_probs,
        step_dones,
        step_invalid_actions_list,
        step_actions_onehot,
        discount_list,
        weights,
    ):

        # burnin
        _, hidden_states = model_q_online(*in_burnin, hidden_states)
        _, hidden_states_t = model_q_target(*in_burnin, hidden_states_t)

        # targetQ
        q_target, _ = model_q_target(*in_steps, hidden_states_t)
        q_target = q_target.numpy()

        # --- 勾配 + targetQを計算
        td_errors_list = []
        with tf.GradientTape() as tape:
            q, _ = model_q_online(*in_steps, hidden_states, training=True)
            frozen_q = tf.stop_gradient(q).numpy()

            # 最後は学習しないので除く
            tf.stop_gradient(q[:, -1, :])
            q = q[:, :-1, :]

            # --- TargetQを計算
            target_q_list = []
            for idx in range(self.config.batch_size):
                retrace = 1
                next_td_error = 0
                td_errors = []
                target_q = []

                # 後ろから計算
                for t in reversed(range(self.config.sequence_length)):
                    action = step_actions[idx][t]
                    mu_prob = step_probs[idx][t]
                    reward = step_rewards[idx][t]
                    done = step_dones[idx][t]
                    invalid_actions = step_invalid_actions_list[idx][t]
                    next_invalid_actions = step_invalid_actions_list[idx][t + 1]

                    if done:
                        gain = reward
                    else:
                        # DoubleDQN: indexはonlineQから選び、値はtargetQを選ぶ
                        if self.config.enable_double_dqn:
                            n_q = frozen_q[idx][t + 1]
                        else:
                            n_q = q_target[idx][t + 1]
                        n_q = [(-np.inf if a in next_invalid_actions else v) for a, v in enumerate(n_q)]
                        n_act_idx = np.argmax(n_q)
                        maxq = q_target[idx][t + 1][n_act_idx]
                        if self.config.enable_rescale:
                            maxq = inverse_rescaling(maxq)
                        gain = reward + discount_list[idx] * maxq
                    if self.config.enable_rescale:
                        gain = rescaling(gain)
                    target_q.insert(0, gain + retrace * next_td_error)

                    td_error = gain - frozen_q[idx][t][action]
                    td_errors.append(td_error)
                    if self.config.enable_retrace:
                        # TDerror
                        next_td_error = td_error

                        # retrace
                        pi_probs = calc_epsilon_greedy_probs(
                            frozen_q[idx][t],
                            invalid_actions,
                            0.0,
                            self.config.action_num,
                        )
                        pi_prob = pi_probs[action]
                        _r = self.config.retrace_h * np.minimum(1, pi_prob / mu_prob)
                        retrace *= discount_list[idx] * _r
                target_q_list.append(target_q)
                td_errors_list.append(np.mean(td_errors))
            target_q_list = np.asarray(target_q_list)

            # --- update Q
            q_onehot = tf.reduce_sum(q * step_actions_onehot, axis=2)
            loss = self.q_loss(target_q_list * weights, q_onehot * weights)
            loss += tf.reduce_sum(model_q_online.losses)

        grads = tape.gradient(loss, model_q_online.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_q_online.trainable_variables))

        return td_errors_list, loss.numpy()


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

        self._recent_states = [self.dummy_state for _ in range(self.config.burnin + self.config.sequence_length + 1)]
        self.recent_actions = [
            random.randint(0, self.config.action_num - 1)
            for _ in range(self.config.burnin + self.config.sequence_length + 1)
        ]
        self.recent_probs = [1.0 / self.config.action_num for _ in range(self.config.sequence_length)]
        self.recent_rewards_ext = [0.0 for _ in range(self.config.burnin + self.config.sequence_length + 1)]
        self.recent_rewards_int = [0.0 for _ in range(self.config.burnin + self.config.sequence_length + 1)]
        self.recent_done = [False for _ in range(self.config.sequence_length)]
        self.recent_invalid_actions = [[] for _ in range(self.config.sequence_length + 1)]

        self.hidden_state_ext = self.parameter.q_ext_online.init_hidden_state()
        self.hidden_state_int = self.parameter.q_int_online.init_hidden_state()
        self.recent_hidden_states_ext = [
            [self.hidden_state_ext[0][0], self.hidden_state_ext[1][0]]
            for _ in range(self.config.burnin + self.config.sequence_length + 1)
        ]
        self.recent_hidden_states_int = [
            [self.hidden_state_int[0][0], self.hidden_state_int[1][0]]
            for _ in range(self.config.burnin + self.config.sequence_length + 1)
        ]

        self._recent_states.pop(0)
        self._recent_states.append(state.astype(np.float32))
        self.recent_invalid_actions.pop(0)
        self.recent_invalid_actions.append(invalid_actions)

        # TD誤差を計算するか
        if self.config.memory_name == "ReplayMemory":
            self._calc_td_error = False
        elif not self.distributed:
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
        self.onehot_actor_idx = tf.one_hot(np.array(self.actor_index), self.config.actor_num)
        self.onehot_actor_idx = tf.expand_dims(tf.expand_dims(self.onehot_actor_idx, 0), 0)

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
        prev_onehot_action = tf.one_hot(np.array(self.action), self.config.action_num)
        prev_onehot_action = tf.expand_dims(tf.expand_dims(prev_onehot_action, 0), 0)

        in_ = [
            self._recent_states[-1][np.newaxis, np.newaxis, ...],
            np.array([[[self.reward_ext]]], dtype=np.float32),
            np.array([[[self.reward_int]]], dtype=np.float32),
            prev_onehot_action,
            self.onehot_actor_idx,
        ]
        self.q_ext, self.hidden_state_ext = self.parameter.q_ext_online(*in_, self.hidden_state_ext)
        self.q_int, self.hidden_state_int = self.parameter.q_int_online(*in_, self.hidden_state_int)
        self.q_ext = self.q_ext[0][0].numpy()
        self.q_int = self.q_int[0][0].numpy()
        self.q = self.q_ext + self.beta * self.q_int

        probs = calc_epsilon_greedy_probs(self.q, invalid_actions, self.epsilon, self.config.action_num)
        self.action = random_choice_by_probs(probs)
        self.prob = probs[self.action]
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
            n_s = next_state.astype(np.float32)[np.newaxis, ...]
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

        self._recent_states.pop(0)
        self._recent_states.append(next_state.astype(np.float32))
        self.recent_actions.pop(0)
        self.recent_actions.append(self.action)
        self.recent_probs.pop(0)
        self.recent_probs.append(self.prob)
        self.recent_rewards_ext.pop(0)
        self.recent_rewards_ext.append(reward_ext)
        self.recent_rewards_int.pop(0)
        self.recent_rewards_int.append(self.reward_int)
        self.recent_done.pop(0)
        self.recent_done.append(done)
        self.recent_invalid_actions.pop(0)
        self.recent_invalid_actions.append(next_invalid_actions)
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
                self._recent_states.pop(0)
                self._recent_states.append(self.dummy_state)
                self.recent_actions.pop(0)
                self.recent_actions.append(random.randint(0, self.config.action_num - 1))
                self.recent_probs.pop(0)
                self.recent_probs.append(1.0 / self.config.action_num)
                self.recent_rewards_ext.pop(0)
                self.recent_rewards_ext.append(0.0)
                self.recent_rewards_int.pop(0)
                self.recent_rewards_int.append(0.0)
                self.recent_done.pop(0)
                self.recent_done.append(True)
                self.recent_invalid_actions.pop(0)
                self.recent_invalid_actions.append([])
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
                            td_error = reward_ext - info["q"]
                        else:
                            td_error = (reward_ext + self.beta * reward_int) - info["q"]
                        self.remote_memory.add(batch, td_error)

        return _info

    def _add_memory(self, calc_info):
        batch = {
            "states": self._recent_states[:],
            "actions": self.recent_actions[:],
            "probs": self.recent_probs[:],
            "rewards_ext": self.recent_rewards_ext[:],
            "rewards_int": self.recent_rewards_int[:],
            "dones": self.recent_done[:],
            "actor": self.actor_index,
            "invalid_actions": self.recent_invalid_actions[:],
            "hidden_states_ext": self.recent_hidden_states_ext[0],
            "hidden_states_int": self.recent_hidden_states_int[0],
        }

        if self._calc_td_error:
            # エピソード最後に計算してメモリに送る
            self._history_batch.append([batch, calc_info])
        else:
            # 計算する必要がない場合はそのままメモリに送る
            self.remote_memory.add(batch, None)

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

    def render_terminal(self, env, worker, **kwargs) -> None:
        invalid_actions = self.recent_invalid_actions[-1]
        q_ext = self.q_ext
        q_int = self.q_int
        q = self.q

        print(f"episodic_reward: {self.episodic_reward}")
        print(f"lifelong_reward: {self.lifelong_reward}")
        print(f"reward_int     : {self.reward_int}")

        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            s = f"{q[a]:5.3f}"
            s += f"{a:2d}: {q[a]:5.3f} = {q_ext[a]:5.3f} + {self.beta} * {q_int[a]:5.3f}"
            return s

        render_discrete_action(invalid_actions, maxa, env, _render_sub)
