import logging
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.algorithms.base_dqn import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import RLProcessor
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.base.spaces.space import SpaceBase
from srl.rl import functions as funcs
from srl.rl.functions import inverse_rescaling, rescaling
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig, RLPriorityReplayBuffer
from srl.rl.models.config.input_image_block import InputImageBlockConfig
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig
from srl.rl.schedulers.scheduler import SchedulerConfig
from srl.rl.tf.blocks.alphazero_image_block import AlphaZeroImageBlock
from srl.rl.tf.model import KerasModelAddedSummary

kl = keras.layers
logger = logging.getLogger(__name__)

"""
Paper
https://openreview.net/forum?id=X6D9bAHhBQ1
"""


@dataclass
class Config(RLConfig):
    num_simulations: int = 20
    discount: float = 0.999

    #: <:ref:`InputImageBlockConfig`>
    input_image_block: InputImageBlockConfig = field(default_factory=lambda: InputImageBlockConfig().set_alphazero_block())
    #: Learning rate
    lr: float = 0.001
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(
        default_factory=lambda: (
            LRSchedulerConfig().set_step(100_000, 0.0001)  #
        )
    )

    #: Batch size
    batch_size: int = 32
    #: <:ref:`PriorityReplayBufferConfig`>
    memory: PriorityReplayBufferConfig = field(default_factory=lambda: PriorityReplayBufferConfig())

    # カテゴリ化する範囲
    v_min: int = -10
    v_max: int = 10

    # policyの温度パラメータのリスト
    policy_tau: Optional[float] = None
    #: <:ref:`SchedulerConfig`>
    policy_tau_scheduler: SchedulerConfig = field(
        default_factory=lambda: (
            SchedulerConfig(default_scheduler=True)  #
            .add_constant(1.0, 50_000)
            .add_constant(0.5, 25_000)
            .add_constant(0.25)
        )
    )

    # td_steps: int = 5      # multisteps
    unroll_steps: int = 5  # unroll_steps
    codebook_size: int = 32  # codebook

    # Root prior exploration noise.
    root_dirichlet_alpha: float = 0.3
    root_dirichlet_fraction: float = 0.1
    root_dirichlet_adaptive: bool = False

    # PUCT
    c_base: float = 19652
    c_init: float = 1.25

    # model
    dynamics_blocks: int = 15
    commitment_cost: float = 0.25  # VQ_VAEのβ
    weight_decay: float = 0.0001
    weight_decay_afterstate: float = 0.001  # 強めに掛けたほうが安定する気がする

    # rescale
    enable_rescale: bool = True

    def get_name(self) -> str:
        return "StochasticMuZero"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if not prev_observation_space.is_image():
            raise ValueError(f"The input supports only image format. {prev_observation_space}")
        return self.input_image_block.get_processors()

    def get_framework(self) -> str:
        return "tensorflow"

    def validate_params(self) -> None:
        super().validate_params()
        if not (self.v_min < self.v_max):
            raise ValueError(f"assert {self.v_min} < {self.v_max}")
        if not (self.unroll_steps > 0):
            raise ValueError(f"assert {self.unroll_steps} > 0")


register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


class Memory(RLPriorityReplayBuffer):
    def setup(self) -> None:
        self.q_min = float("inf")
        self.q_max = float("-inf")
        self.register_worker_func(self.add_q, lambda x1, x2: (x1, x2))
        self.register_trainer_recv_func(self.get_q)

    def add_q(self, q_min, q_max, serialized: bool = False):
        self.q_min = min(self.q_min, q_min)
        self.q_max = max(self.q_max, q_max)

    def get_q(self):
        return self.q_min, self.q_max


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _RepresentationNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config):
        super().__init__()

        self.in_block = config.input_image_block.create_tf_block(
            config.observation_space,
            out_flatten=False,
        )

        # build & 出力shapeを取得
        dummy_state = np.zeros(shape=(1,) + config.observation_space.shape, dtype=np.float32)
        hidden_state = self(dummy_state)
        self.hidden_state_shape = hidden_state.shape[1:]  # type:ignore , ignore check "None"

    def call(self, state, training=False):
        x = self.in_block(state, training=training)

        # 隠れ状態はアクションとスケールを合わせるため0-1で正規化(一応batch毎)
        batch, h, w, d = x.shape
        if batch is None:
            return x
        s_min = tf.reduce_min(tf.reshape(x, (batch, -1)), axis=1, keepdims=True)
        s_max = tf.reduce_max(tf.reshape(x, (batch, -1)), axis=1, keepdims=True)
        s_min = s_min * tf.ones((batch, h * w * d), dtype=tf.float32)
        s_max = s_max * tf.ones((batch, h * w * d), dtype=tf.float32)
        s_min = tf.reshape(s_min, (batch, h, w, d))
        s_max = tf.reshape(s_max, (batch, h, w, d))
        epsilon = 1e-4  # div0 回避
        x = (x - s_min + epsilon) / tf.maximum((s_max - s_min), 2 * epsilon)

        return x


class _DynamicsNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, as_shape):
        super().__init__()
        self.c_size = config.codebook_size
        v_num = config.v_max - config.v_min + 1
        h, w, ch = as_shape

        # --- hidden_state
        self.image_block = AlphaZeroImageBlock(
            n_blocks=config.dynamics_blocks,
            filters=ch,
            # l2=config.weight_decay, TODO
        )

        # reward
        self.reward_layers = [
            kl.Conv2D(
                1,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Flatten(),
            kl.Dense(
                v_num,
                activation="softmax",
                kernel_initializer="truncated_normal",
                bias_initializer="truncated_normal",
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
        ]

        # build
        self._in_shape = (h, w, ch + self.c_size)
        self(np.zeros((1,) + self._in_shape))

    def call(self, in_state, training=False):
        # hidden state
        x = self.image_block(in_state, training=training)

        # reward
        reward_category = in_state
        for layer in self.reward_layers:
            reward_category = layer(reward_category, training=training)

        return x, reward_category

    def predict(self, as_state, code_state, training=False):
        batch_size, h, w, _ = as_state.shape

        code_image = tf.repeat(code_state, h * w, axis=1)  # (b, c_size)->(b, c_size * h * w)
        code_image = tf.reshape(code_image, (batch_size, self.c_size, h, w))  # (b, c_size * h * w)->(b, c_size, h, w)
        code_image = tf.transpose(code_image, perm=[0, 2, 3, 1])  # (b, c_size, h, w)->(b, h, w, c_size)

        in_state = tf.concat([as_state, code_image], axis=3)
        x, reward_category = self.call(in_state, training=training)

        # 隠れ状態はアクションとスケールを合わせるため0-1で正規化(一応batch毎)
        batch, h, w, d = x.shape
        s_min = tf.reduce_min(tf.reshape(x, (batch, -1)), axis=1, keepdims=True)
        s_max = tf.reduce_max(tf.reshape(x, (batch, -1)), axis=1, keepdims=True)
        s_min = s_min * tf.ones((batch, h * w * d), dtype=tf.float32)
        s_max = s_max * tf.ones((batch, h * w * d), dtype=tf.float32)
        s_min = tf.reshape(s_min, (batch, h, w, d))
        s_max = tf.reshape(s_max, (batch, h, w, d))
        epsilon = 1e-4  # div0 回避
        x = (x - s_min + epsilon) / tf.maximum((s_max - s_min), 2 * epsilon)

        return x, reward_category


class _PredictionNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, hidden_shape):
        super().__init__()
        v_num = config.v_max - config.v_min + 1

        # --- policy
        self.policy_layers = [
            kl.Conv2D(
                2,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Flatten(),
            kl.Dense(
                config.action_space.n,
                activation="softmax",
                kernel_initializer="truncated_normal",
                bias_initializer="truncated_normal",
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
        ]

        # --- value
        self.value_layers = [
            kl.Conv2D(
                1,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Flatten(),
            kl.Dense(
                v_num,
                activation="softmax",
                kernel_initializer="truncated_normal",
                bias_initializer="truncated_normal",
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
        ]

        # build
        self(np.zeros((1,) + hidden_shape))

    def call(self, state, training=False):
        policy = state
        for layer in self.policy_layers:
            policy = layer(policy, training=training)

        value = state
        for layer in self.value_layers:
            value = layer(value, training=training)

        return policy, value


class _AfterstateDynamicsNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, hidden_shape):
        super().__init__()
        self.action_num = config.action_space.n
        h, w, ch = hidden_shape

        self.image_block = AlphaZeroImageBlock(
            n_blocks=config.dynamics_blocks,
            filters=ch,
            # l2=config.weight_decay_afterstate,  TODO
        )

        # build
        self(np.zeros((1, h, w, ch + self.action_num)))

    def call(self, in_state, training=False):
        return self.image_block(in_state, training=training)

    def predict(self, hidden_state, action, training=False):
        batch_size, h, w, _ = hidden_state.shape

        action_image = tf.one_hot(action, self.action_num)  # (batch, action)
        action_image = tf.repeat(action_image, repeats=h * w, axis=1)  # (batch, action * h * w)
        action_image = tf.reshape(action_image, (batch_size, self.action_num, h, w))  # (batch, action, h, w)
        action_image = tf.transpose(action_image, perm=[0, 2, 3, 1])  # (batch, h, w, action)

        # hidden_state + action_space
        in_state = tf.concat([hidden_state, action_image], axis=3)
        return self.call(in_state, training=training)


class _AfterstatePredictionNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, as_shape):
        super().__init__()
        v_num = config.v_max - config.v_min + 1

        # --- code
        self.code_layers = [
            kl.Conv2D(
                2,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Flatten(),
            kl.Dense(
                config.codebook_size,
                activation="softmax",
                kernel_initializer="truncated_normal",
                bias_initializer="truncated_normal",
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
        ]

        # --- Q
        self.q_layers = [
            kl.Conv2D(
                1,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Flatten(),
            kl.Dense(
                v_num,
                activation="softmax",
                kernel_initializer="truncated_normal",
                bias_initializer="truncated_normal",
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
        ]
        # build
        self(np.zeros((1,) + as_shape))

    def call(self, state, training=False):
        code = state
        for layer in self.code_layers:
            code = layer(code, training=training)

        q = state
        for layer in self.q_layers:
            q = layer(q, training=training)

        return code, q


class _VQVAE(KerasModelAddedSummary):
    def __init__(self, config: Config):
        super().__init__()

        # --- codebook(one-hot vector)
        self.c_size = config.codebook_size
        self.codebook = np.identity(self.c_size, dtype=np.float32)[np.newaxis, ...]

        self.in_block = config.input_image_block.create_tf_block(
            config.observation_space,
            out_flatten=False,
        )

        self.out_layers = [
            kl.Conv2D(
                filters=2,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Flatten(),
            kl.Dense(
                config.codebook_size,
                activation="softmax",
                kernel_initializer="truncated_normal",
                bias_initializer="truncated_normal",
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
        ]

        # build
        self._in_shape = config.observation_space.shape
        self(np.zeros((1,) + self._in_shape))

    def call(self, state, training=False):
        x = self.in_block(state, training=training)
        for layer in self.out_layers:
            x = layer(x, training=training)
        if state.shape[0] is None:
            return x
        return self.encode(x), x

    def encode(self, x):
        # codebookから変換、とりあえず愚直に
        batch = x.shape[0]
        codebook = np.tile(self.codebook, (batch, 1, 1))  # [1, c, c]->[b, c, c]

        x = np.tile(x, (1, self.c_size))  # [b, c]->[b, c*c]
        x = np.reshape(x, (-1, self.c_size, self.c_size))  # [b, c*c]->[b, c, c]

        distance = np.sum((x - codebook) ** 2, axis=2)
        indices = np.argmin(distance, axis=1)
        onehot = np.identity(self.c_size, dtype=np.float32)[indices]
        onehot = np.tile(onehot, (1, self.c_size)).reshape((-1, self.c_size, self.c_size))  # [b, c, c]
        code = np.sum(onehot * codebook, axis=2)  # [b, c, c]->[b, c]

        return code


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def setup(self) -> None:
        self.representation_network = _RepresentationNetwork(self.config)
        # 出力shapeを取得
        hidden_state_shape = self.representation_network.hidden_state_shape

        self.dynamics_network = _DynamicsNetwork(self.config, hidden_state_shape)
        self.prediction_network = _PredictionNetwork(self.config, hidden_state_shape)
        self.afterstate_dynamics_network = _AfterstateDynamicsNetwork(self.config, hidden_state_shape)
        self.afterstate_prediction_network = _AfterstatePredictionNetwork(self.config, hidden_state_shape)
        self.vq_vae = _VQVAE(self.config)

        self.q_min = np.inf
        self.q_max = -np.inf

        # cache用 (simulationで何回も使うので)
        self.P = {}
        self.V = {}
        self.C = {}
        self.Q = {}

    def call_restore(self, data: Any, **kwargs) -> None:
        self.prediction_network.set_weights(data[0])
        self.dynamics_network.set_weights(data[1])
        self.representation_network.set_weights(data[2])
        self.afterstate_dynamics_network.set_weights(data[3])
        self.afterstate_prediction_network.set_weights(data[4])
        self.vq_vae.set_weights(data[5])
        self.q_min = data[6]
        self.q_max = data[7]
        self.reset_cache()

    def call_backup(self, **kwargs):
        return [
            self.prediction_network.get_weights(),
            self.dynamics_network.get_weights(),
            self.representation_network.get_weights(),
            self.afterstate_dynamics_network.get_weights(),
            self.afterstate_prediction_network.get_weights(),
            self.vq_vae.get_weights(),
            self.q_min,
            self.q_max,
        ]

    def summary(self, **kwargs):
        self.representation_network.summary(**kwargs)
        self.dynamics_network.summary(**kwargs)
        self.prediction_network.summary(**kwargs)
        self.afterstate_dynamics_network.summary(**kwargs)
        self.afterstate_prediction_network.summary(**kwargs)
        self.vq_vae.summary(**kwargs)

    # ------------------------

    def prediction(self, state, state_str):
        if state_str not in self.P:
            p, v_category = self.prediction_network(state)  # type:ignore , ignore check "None"
            self.P[state_str] = p[0].numpy()
            self.V[state_str] = funcs.twohot_decode(
                v_category.numpy()[0],
                abs(self.config.v_max - self.config.v_min) + 1,
                self.config.v_min,
                self.config.v_max,
            )

    def afterstate_prediction(self, as_state, state_str):
        if state_str not in self.C:
            c, q_category = self.afterstate_prediction_network(as_state)  # type:ignore , ignore check "None"
            self.Q[state_str] = funcs.twohot_decode(
                q_category.numpy()[0],
                abs(self.config.v_max - self.config.v_min) + 1,
                self.config.v_min,
                self.config.v_max,
            )
            self.C[state_str] = self.vq_vae.encode(c)

    def reset_cache(self):
        self.P = {}
        self.V = {}
        self.C = {}
        self.Q = {}


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
def _scale_gradient(tensor, scale):
    """muzeroより流用"""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.opt_rep = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))
        self.opt_pre = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))
        self.opt_dyn = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))
        self.opt_after_dyn = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))
        self.opt_after_pre = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))
        self.opt_vq_vae = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))

    def _cross_entropy_loss(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-6, y_pred)  # log(0)回避用
        loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1)
        return loss

    def _mse_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return
        batches, weights, update_args = batches

        # (batch, dict, steps, val) -> (steps, batch, val)
        states_list = []
        actions_list = []
        policies_list = []
        values_list = []
        rewards_list = []
        for i in range(self.config.unroll_steps + 1):
            states = []
            actions = []
            policies = []
            values = []
            rewards = []
            for b in batches:
                states.append(b["states"][i])
                policies.append(b["policies"][i])
                values.append(b["values"][i])
                if i < self.config.unroll_steps:
                    actions.append(b["actions"][i])
                    rewards.append(b["rewards"][i])
            states_list.append(np.asarray(states))
            actions_list.append(actions)
            policies_list.append(np.asarray(policies).astype(np.float32))
            values_list.append(np.asarray(values).astype(np.float32))
            rewards_list.append(np.asarray(rewards).astype(np.float32))

        with tf.GradientTape() as tape:
            # --- 1st step
            hidden_states = self.parameter.representation_network(states_list[0], training=True)
            p_pred, v_pred = self.parameter.prediction_network(hidden_states, training=True)

            # loss
            policy_loss = self._cross_entropy_loss(policies_list[0], p_pred)
            v_loss = self._cross_entropy_loss(values_list[0], v_pred)
            reward_loss = tf.constant([0] * self.config.batch_size, dtype=tf.float32)
            chance_loss = tf.constant([0] * self.config.batch_size, dtype=tf.float32)
            q_loss = tf.constant([0] * self.config.batch_size, dtype=tf.float32)
            vae_loss = tf.constant([0] * self.config.batch_size, dtype=tf.float32)

            # --- unroll steps
            gradient_scale = 1 / self.config.unroll_steps
            for t in range(self.config.unroll_steps):
                after_states = self.parameter.afterstate_dynamics_network.predict(hidden_states, actions_list[t], training=True)
                chance_pred, q_pred = self.parameter.afterstate_prediction_network(after_states, training=True)
                chance_code, chance_vae_pred = self.parameter.vq_vae(states_list[t + 1], training=True)

                chance_loss += _scale_gradient(self._cross_entropy_loss(chance_code, chance_pred), gradient_scale)
                q_loss += _scale_gradient(self._cross_entropy_loss(values_list[t], q_pred), gradient_scale)
                vae_loss += _scale_gradient(tf.reduce_mean(tf.square(chance_code - chance_vae_pred), axis=1), gradient_scale)

                hidden_states, rewards_pred = self.parameter.dynamics_network.predict(after_states, chance_code, training=True)
                p_pred, v_pred = self.parameter.prediction_network(hidden_states)

                # 安定しなかったのでMSEに変更
                policy_loss += _scale_gradient(self._mse_loss(policies_list[t + 1], p_pred), gradient_scale)
                v_loss += _scale_gradient(self._mse_loss(values_list[t + 1], v_pred), gradient_scale)
                reward_loss += _scale_gradient(self._cross_entropy_loss(rewards_list[t], rewards_pred), gradient_scale)

                hidden_states = _scale_gradient(hidden_states, 0.5)

            loss = v_loss + policy_loss + reward_loss + chance_loss + q_loss + self.config.commitment_cost * vae_loss
            loss = tf.reduce_mean(loss * weights)

            # 各ネットワークの正則化項を加える
            loss += tf.reduce_sum(self.parameter.representation_network.losses)
            loss += tf.reduce_sum(self.parameter.prediction_network.losses)
            loss += tf.reduce_sum(self.parameter.dynamics_network.losses)
            loss += tf.reduce_sum(self.parameter.afterstate_dynamics_network.losses)
            loss += tf.reduce_sum(self.parameter.afterstate_prediction_network.losses)
            loss += tf.reduce_sum(self.parameter.vq_vae.losses)

        variables = [
            self.parameter.representation_network.trainable_variables,
            self.parameter.dynamics_network.trainable_variables,
            self.parameter.prediction_network.trainable_variables,
            self.parameter.afterstate_dynamics_network.trainable_variables,
            self.parameter.afterstate_prediction_network.trainable_variables,
            self.parameter.vq_vae.trainable_variables,
        ]
        grads = tape.gradient(loss, variables)
        self.opt_rep.apply_gradients(zip(grads[0], variables[0]))
        self.opt_dyn.apply_gradients(zip(grads[1], variables[1]))
        self.opt_pre.apply_gradients(zip(grads[2], variables[2]))
        self.opt_after_dyn.apply_gradients(zip(grads[3], variables[3]))
        self.opt_after_pre.apply_gradients(zip(grads[4], variables[4]))
        self.opt_vq_vae.apply_gradients(zip(grads[5], variables[5]))

        self.train_count += 1
        self.info["loss"] = loss.numpy()
        self.info["v_loss"] = np.mean(v_loss.numpy())
        self.info["policy_loss"] = np.mean(policy_loss.numpy())
        self.info["reward_loss"] = np.mean(reward_loss.numpy())
        self.info["chance_loss"] = np.mean(chance_loss.numpy())
        self.info["q_loss"] = np.mean(q_loss.numpy())
        self.info["vae_loss"] = np.mean(vae_loss.numpy())

        # memory update
        priorities = np.abs(v_loss.numpy())
        self.memory.update(update_args, priorities, self.train_count)

        # 学習したらキャッシュは削除
        self.parameter.reset_cache()

        # --- 正規化用Qを保存(parameterはtrainerからしか保存されない)
        # (remote_memory -> trainer -> parameter)
        q_min, q_max = self.memory.get_q()
        self.parameter.q_min = q_min
        self.parameter.q_max = q_max


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker[Config, Parameter, Memory]):
    def on_setup(self, worker, context) -> None:
        self.env_player_num = worker.env.player_num
        self.policy_tau_sch = self.config.policy_tau_scheduler.create(self.config.policy_tau)
        self._v_min = np.inf
        self._v_max = -np.inf

    def on_reset(self, worker):
        self.history = []
        self.episode_history = []

        self.N = {}  # 訪問回数(s,a)
        self.W = {}  # 累計報酬(s,a)
        self.Q = {}  # 報酬(s,a)

    def _init_state(self, state_str, num):
        if state_str not in self.N:
            self.N[state_str] = [0 for _ in range(num)]
            self.W[state_str] = [0 for _ in range(num)]
            self.Q[state_str] = [0 for _ in range(num)]

    def policy(self, worker) -> int:
        invalid_actions = worker.invalid_actions

        # --- シミュレーションしてpolicyを作成
        self.s0 = self.parameter.representation_network(worker.state[np.newaxis, ...])
        self.s0_str = self.s0.ref()  # type:ignore , ignore check "None"
        for _ in range(self.config.num_simulations):
            self._simulation(self.s0, self.s0_str, invalid_actions, is_afterstate=False)

        # 正規化用Qを保存できるように送信(remote_memory -> trainer -> parameter)
        self.memory.add_q(self.parameter.q_min, self.parameter.q_max)

        # V
        self.state_v = self.parameter.V[self.s0_str]

        # --- 確率に比例したアクションを選択
        if not self.training:
            policy_tau = 0  # 評価時は決定的に
        else:
            policy_tau = self.policy_tau_sch.update(self.step_in_training).to_float()

        if policy_tau == 0:
            counts = np.asarray(self.N[self.s0_str])
            action = np.random.choice(np.where(counts == counts.max())[0])
        else:
            step_policy = np.array([self.N[self.s0_str][a] ** (1 / policy_tau) for a in range(self.config.action_space.n)])
            step_policy /= step_policy.sum()
            action = funcs.random_choice_by_probs(step_policy)

        # 学習用のpolicyはtau=1
        N = sum(self.N[self.s0_str])
        self.step_policy = [self.N[self.s0_str][a] / N for a in range(self.config.action_space.n)]

        self.action = int(action)
        return self.action

    def _simulation(self, state, state_str, invalid_actions, is_afterstate, depth: int = 0):
        if depth >= 99999:  # for safety
            return 0

        if not is_afterstate:
            self._init_state(state_str, self.config.action_space.n)
            self.parameter.prediction(state, state_str)

            # actionを選択
            puct_list = self._calc_puct(state_str, invalid_actions, depth == 0)
            action = np.random.choice(np.where(puct_list == np.max(puct_list))[0])

            # 次の状態を取得(after state)
            n_state = self.parameter.afterstate_dynamics_network.predict(state, [action])
            reward = 0
            is_afterstate = True

        else:
            self._init_state(state_str, self.config.codebook_size)
            self.parameter.afterstate_prediction(state, state_str)
            c = self.parameter.C[state_str]
            action = np.argmax(c[0])  # outcomes

            # 次の状態を取得
            n_state, reward_category = self.parameter.dynamics_network.predict(state, c)
            reward = funcs.twohot_decode(
                reward_category.numpy()[0],
                abs(self.config.v_max - self.config.v_min) + 1,
                self.config.v_min,
                self.config.v_max,
            )
            is_afterstate = False

        n_state_str = n_state.ref()
        enemy_turn = self.env_player_num > 1  # 2player以上は相手番と決め打ち

        if self.N[state_str][action] == 0:
            # leaf node ならロールアウト
            if is_afterstate:
                self.parameter.afterstate_prediction(n_state, n_state_str)
                n_value = self.parameter.Q[n_state_str]
            else:
                self.parameter.prediction(n_state, n_state_str)
                n_value = self.parameter.V[n_state_str]

        else:
            # 子ノードに降りる(展開)
            n_value = self._simulation(n_state, n_state_str, [], is_afterstate, depth + 1)

        # 次が相手のターンなら、報酬は最小になってほしいので-をかける
        if enemy_turn:
            n_value = -n_value

        # 割引報酬
        reward = reward + self.config.discount * n_value

        self.N[state_str][action] += 1
        self.W[state_str][action] += reward
        self.Q[state_str][action] = self.W[state_str][action] / self.N[state_str][action]

        self.parameter.q_min = min(self.parameter.q_min, self.Q[state_str][action])
        self.parameter.q_max = max(self.parameter.q_max, self.Q[state_str][action])

        return reward

    def _calc_puct(self, state_str, invalid_actions, is_root):
        # ディリクレノイズ
        if is_root:
            dir_alpha = self.config.root_dirichlet_alpha
            if self.config.root_dirichlet_adaptive:
                dir_alpha = 1.0 / np.sqrt(self.config.action_space.n - len(invalid_actions))
            noises = np.random.dirichlet([dir_alpha] * self.config.action_space.n)
        else:
            noises = []

        N = np.sum(self.N[state_str])
        scores = np.zeros(self.config.action_space.n)
        for a in range(self.config.action_space.n):
            if a in invalid_actions:
                score = -np.inf
            else:
                # P(s,a): 過去のMCTSの結果を教師あり学習した結果
                # U(s,a) = C(s) * P(s,a) * sqrt(N(s)) / (1+N(s,a))
                # C(s) = log((1+N(s)+base)/base) + c_init
                # score = Q(s,a) + U(s,a)
                P = self.parameter.P[state_str][a]

                # rootはディリクレノイズを追加
                if is_root:
                    e = self.config.root_dirichlet_fraction
                    P = (1 - e) * P + e * noises[a]

                n = self.N[state_str][a]
                c = np.log((1 + N + self.config.c_base) / self.config.c_base) + self.config.c_init
                u = c * P * (np.sqrt(N) / (1 + n))
                q = self.Q[state_str][a]

                # 過去観測したQ値で正規化(MinMax)
                if self.parameter.q_min < self.parameter.q_max:
                    q = (q - self.parameter.q_min) / (self.parameter.q_max - self.parameter.q_min)

                score = q + u
                if np.isnan(score):
                    logger.warning(
                        "puct score is nan. action={}, score={}, q={}, u={}, Q={}, P={}".format(
                            a,
                            score,
                            q,
                            u,
                            self.Q[state_str],
                            self.parameter.P[state_str],
                        )
                    )
                    score = -np.inf

            scores[a] = score
        return scores

    def on_step(self, worker):
        if not self.training:
            return

        self.history.append(
            {
                "state": worker.state,
                "action": self.action,
                "policy": self.step_policy,
                "reward": worker.reward,
                "state_v": self.state_v,
            }
        )

        if worker.done:
            zero_category = funcs.twohot_encode(0, abs(self.config.v_max - self.config.v_min) + 1, self.config.v_min, self.config.v_max)
            zero_state = np.zeros(self.config.observation_space.shape)

            # calc MC reward
            reward = 0
            for h in reversed(self.history):
                reward = h["reward"] + self.config.discount * reward
                h["discount_reward"] = reward

            # batch create
            for idx in range(len(self.history)):
                # --- policies
                policies = [[1 / self.config.action_space.n] * self.config.action_space.n for _ in range(self.config.unroll_steps + 1)]
                for i in range(self.config.unroll_steps + 1):
                    if idx + i >= len(self.history):
                        break
                    policies[i] = self.history[idx + i]["policy"]

                # --- values
                values = [zero_category for _ in range(self.config.unroll_steps + 1)]
                priority = 0
                for i in range(self.config.unroll_steps + 1):
                    if idx + i >= len(self.history):
                        break
                    v = self.history[idx + i]["discount_reward"]
                    if self.config.enable_rescale:
                        v = rescaling(v)
                    priority += v - self.history[idx + i]["state_v"]
                    self._v_min = min(self._v_min, v)
                    self._v_max = max(self._v_max, v)
                    values[i] = funcs.twohot_encode(v, abs(self.config.v_max - self.config.v_min) + 1, self.config.v_min, self.config.v_max)
                priority /= self.config.unroll_steps + 1

                # --- states
                states = [zero_state for _ in range(self.config.unroll_steps + 1)]
                for i in range(self.config.unroll_steps + 1):
                    if idx + i >= len(self.history):
                        break
                    states[i] = self.history[idx + i]["state"]

                # --- actions
                actions = [random.randint(0, self.config.action_space.n - 1) for _ in range(self.config.unroll_steps)]
                for i in range(self.config.unroll_steps):
                    if idx + i >= len(self.history):
                        break
                    actions[i] = self.history[idx + i]["action"]

                # --- reward
                rewards = [zero_category for _ in range(self.config.unroll_steps)]
                for i in range(self.config.unroll_steps):
                    if idx + i >= len(self.history):
                        break
                    r = self.history[idx + i]["reward"]
                    if self.config.enable_rescale:
                        r = rescaling(r)
                    self._v_min = min(self._v_min, r)
                    self._v_max = max(self._v_max, r)
                    rewards[i] = funcs.twohot_encode(r, abs(self.config.v_max - self.config.v_min) + 1, self.config.v_min, self.config.v_max)

                self.memory.add(
                    {
                        "states": states,
                        "actions": actions,
                        "policies": policies,
                        "values": values,
                        "rewards": rewards,
                    },
                    priority,
                )

        self.info["v_min"] = self._v_min
        self.info["v_max"] = self._v_max

    def render_terminal(self, worker, **kwargs) -> None:
        self._init_state(self.s0_str, self.config.action_space.n)
        self.parameter.prediction(self.s0, self.s0_str)
        puct = self._calc_puct(self.s0_str, worker.invalid_actions, False)
        maxa = self.action

        v = self.parameter.V[self.s0_str]
        if self.config.enable_rescale:
            v = inverse_rescaling(v)

        print(f"V: {v:.5f}")

        def _render_sub(a: int) -> str:
            after_state = self.parameter.afterstate_dynamics_network.predict(self.s0, [a])
            c, q_category = self.parameter.afterstate_prediction_network(after_state)
            q = funcs.twohot_decode(
                q_category.numpy()[0],
                abs(self.config.v_max - self.config.v_min) + 1,
                self.config.v_min,
                self.config.v_max,
            )
            c = self.parameter.vq_vae.encode(c)
            _, reward_category = self.parameter.dynamics_network.predict(after_state, c)
            reward = funcs.twohot_decode(
                reward_category.numpy()[0],
                abs(self.config.v_max - self.config.v_min) + 1,
                self.config.v_min,
                self.config.v_max,
            )

            if self.config.enable_rescale:
                q = inverse_rescaling(q)
                reward = inverse_rescaling(reward)

            s = f"{self.step_policy[a] * 100:5.1f}%"
            s += f" {self.N[self.s0_str][a]:7d}(N)"
            s += f" {self.Q[self.s0_str][a]:9.5f}(Q)"
            s += f", {puct[a]:9.5f}(PUCT)"
            s += f", {self.parameter.P[self.s0_str][a]:9.5f}(P)"

            s += f", {q:9.5f}(Q_pred)"
            s += f", {np.argmax(c[0]):3d}(code)"
            s += f", {reward:9.5f}(reward)"
            return s

        worker.print_discrete_action_info(int(maxa), _render_sub)
