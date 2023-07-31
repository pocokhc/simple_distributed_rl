import collections
import random
from dataclasses import dataclass
from functools import reduce
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import EnvObservationTypes, RLTypes
from srl.base.rl.algorithms.discrete_action import DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLRemoteMemory, RLTrainer
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.rl.registration import register
from srl.base.rl.worker_run import WorkerRun
from srl.rl.models.tf.input_block import InputBlock
from srl.rl.processors.image_processor import ImageProcessor

kl = keras.layers

"""
vae ref: https://developers-jp.googleblog.com/2019/04/tensorflow-probability-vae.html
ref: https://github.com/zacwellmer/WorldModels
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig):
    train_mode: int = 1

    lr: float = 0.001
    batch_size: int = 32
    capacity: int = 100_000
    memory_warmup_size: int = 100

    # VAE
    z_size: int = 32
    kl_tolerance: float = 0.5

    # MDN-RNN
    sequence_length: int = 10
    rnn_units: int = 256
    num_mixture: int = 5  # number of mixtures in MDN
    temperature: float = 1.15

    # GA
    num_simulations: int = 16
    num_individual: int = 16
    mutation: float = 0.01
    randn_sigma: float = 1.0
    blx_a: float = 0.1

    # other
    dummy_state_val: float = 0.0

    def set_processor(self) -> List[Processor]:
        return [
            ImageProcessor(
                image_type=EnvObservationTypes.COLOR,
                resize=(64, 64),
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
        return "tensorflow"

    def getName(self) -> str:
        return "WorldModels"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.memory_warmup_size < self.capacity
        assert self.batch_size < self.memory_warmup_size
        assert self.temperature >= 0

    @property
    def info_types(self) -> dict:
        return {
            "vae_loss": {},
            "rc_loss": {},
            "kl_loss": {},
            "rnn_loss": {},
        }


register(
    Config(),
    __name__ + ":RemoteMemory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


# ------------------------------------------------------
# RemoteMemory
# ------------------------------------------------------
class RemoteMemory(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.vae_buffer = collections.deque(maxlen=self.config.capacity)
        self.rnn_buffer = collections.deque(maxlen=self.config.capacity)

        self.c_params = None
        self.c_score = -np.inf

    def length(self) -> int:
        return len(self.vae_buffer) + len(self.rnn_buffer)

    def call_restore(self, data: Any, **kwargs) -> None:
        self.vae_buffer = data[0]
        self.rnn_buffer = data[1]

    def call_backup(self, **kwargs):
        return [
            self.vae_buffer,
            self.rnn_buffer,
        ]

    # ---------------------------
    def vae_length(self) -> int:
        return len(self.vae_buffer)

    def vae_add(self, batch: Any):
        self.vae_buffer.append(batch)

    def vae_sample(self, batch_size: int) -> List[Any]:
        return random.sample(self.vae_buffer, batch_size)

    def rnn_length(self) -> int:
        return len(self.rnn_buffer)

    def rnn_add(self, batch: Any):
        self.rnn_buffer.append(batch)

    def rnn_sample(self, batch_size: int) -> List[Any]:
        return random.sample(self.rnn_buffer, batch_size)

    def c_update(self, params, score):
        if self.c_score < score:
            self.c_score = score
            self.c_params = params

    def c_get(self):
        return self.c_params, self.c_score


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _VAE(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        self.z_size = config.z_size
        self.kl_tolerance = config.kl_tolerance

        # --- encoder
        self.encoder_in_block = InputBlock(config.observation_shape, config.env_observation_type)
        self.use_image_head = self.encoder_in_block.use_image_layer
        if self.use_image_head:
            assert config.window_length == 1
            self.encoder_in_layers = [
                kl.Conv2D(filters=32, kernel_size=4, strides=2, activation="relu"),
                kl.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu"),
                kl.Conv2D(filters=128, kernel_size=4, strides=2, activation="relu"),
                kl.Conv2D(filters=256, kernel_size=4, strides=2, activation="relu"),
                kl.Flatten(),
            ]
        else:
            self.encoder_in_layers = [
                kl.Dense(256, activation="relu"),
                kl.Dense(256, activation="relu"),
            ]
        self.encoder_z_mean_layer = kl.Dense(config.z_size)
        self.encoder_z_log_stddev = kl.Dense(config.z_size)

        # --- decoder
        if self.use_image_head:
            self.decoder_in_layers = [
                kl.Dense(2 * 2 * 256, activation="relu"),
                kl.Reshape((1, 1, 2 * 2 * 256)),
                kl.Conv2DTranspose(128, kernel_size=5, strides=2, padding="valid", activation="relu"),
                kl.Conv2DTranspose(64, kernel_size=5, strides=2, padding="valid", activation="relu"),
                kl.Conv2DTranspose(32, kernel_size=6, strides=2, padding="valid", activation="relu"),
                kl.Conv2DTranspose(3, kernel_size=6, strides=2, padding="valid", activation="sigmoid"),
            ]
        else:
            flatten_shape = np.zeros(config.observation_shape).flatten().shape
            self.decoder_in_layers = [
                kl.Dense(256, activation="relu"),
                kl.Dense(256, activation="relu"),
                kl.Dense(flatten_shape[0]),
                kl.Reshape(config.observation_shape),
            ]

        # build
        self.build((None,) + config.observation_shape)

    def call(self, x, training=False):
        return self.decode(self.encode(x, training), training)

    def encode(self, x, training=False):
        x = self.encoder_in_block(x, training=training)
        for layer in self.encoder_in_layers:
            x = layer(x, training=training)
        z_mean = self.encoder_z_mean_layer(x, training=training)
        z_log_stddev = self.encoder_z_log_stddev(x, training=training)

        if x.shape[0] is None:  # type:ignore , ignore check "None"
            return z_mean

        # reparameterize
        e = tf.random.normal(z_mean.shape)  # type:ignore , ignore check "None"
        z = z_mean + tf.exp(0.5 * z_log_stddev) * e  # type:ignore , ignore check "None"

        if training:
            return z_mean, z_log_stddev, z
        else:
            return z

    def decode(self, x, training=False):
        for layer in self.decoder_in_layers:
            x = layer(x, training=training)
        return x

    def sample(self, size=1):
        z = np.random.normal(size=(size, self.z_size))
        return self.decode(z), z

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def summary(self, name: str = "", **kwargs):
        if hasattr(self.encoder_in_block, "init_model_graph"):
            self.encoder_in_block.init_model_graph()

        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x), name=name)
        model.summary(**kwargs)


class _MDNRNN(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        self.action_num = config.action_num
        self.z_size = config.z_size
        self.num_mixture = config.num_mixture
        self.temperature = config.temperature

        # --- LSTM
        self.lstm_layer = kl.LSTM(config.rnn_units, return_sequences=True, return_state=True)

        # --- MDN
        self.mdn_layer = kl.Dense(config.z_size * config.num_mixture * 3)

        # 重みを初期化
        dummy_z = np.zeros(shape=(1, 1, config.z_size), dtype=np.float32)
        dummy_onehot_action = np.zeros(shape=(1, 1, config.action_num), dtype=np.float32)
        self(dummy_z, dummy_onehot_action, None, return_rnn_only=False, training=False)

    def call(self, z, onehot_actions, hidden_state, return_rnn_only, training):
        batch_size = z.shape[0]
        timesteps = z.shape[1]

        # (batch, timesteps, z + action) -> (batch, timesteps, lstm_dim)
        x = tf.concat([z, onehot_actions], axis=2)
        x, h, c = self.lstm_layer(
            x, initial_state=hidden_state, training=training
        )  # type:ignore , ignore check "None"
        if return_rnn_only:
            return [h, c]

        # -> (batch * timesteps, lstm_dim)
        x = tf.reshape(x, (batch_size * timesteps, -1))

        # -> (batch * timesteps, z * num_mix * 3)
        x = self.mdn_layer(x, training=training)

        # -> (batch * timesteps, z, num_mix * 3)
        x = tf.reshape(x, (-1, self.z_size, self.num_mixture * 3))

        # -> (batch * timesteps, z, num_mix) * 3
        pi, mu, log_sigma = tf.split(x, 3, axis=2)

        return pi, mu, log_sigma, [h, c]

    def forward(self, z, action, hidden_state, return_rnn_only):
        assert z.shape[0] == 1
        onehot_actions = tf.one_hot(np.array([action]), self.action_num, axis=1)

        # (batch, shape) -> (batch, 1, shape)
        z = z[:, np.newaxis, ...]
        onehot_actions = onehot_actions[:, np.newaxis, ...]

        return self(z, onehot_actions, hidden_state, return_rnn_only=return_rnn_only, training=False)

    def sample(self, pi, mu, log_sigma):
        batch = pi.shape[0]
        z_size = pi.shape[1]

        sigma = np.exp(log_sigma)

        if self.temperature > 0:
            # softmax
            pi /= self.temperature  # adjust temperatures
            pi = pi - tf.reduce_max(pi, axis=2, keepdims=True)  # overflow_protection
            exp_pi = tf.exp(pi)
            pi = exp_pi / tf.reduce_sum(exp_pi, axis=2, keepdims=True)

        samples = np.zeros((batch, z_size))
        for i in range(batch):
            for j in range(z_size):
                if self.temperature == 0:
                    # 最大値(決定的)
                    idx = np.argmax(pi[i][j])
                    z = mu[i][j][idx]
                else:
                    idx = random.choices([i for i in range(self.num_mixture)], weights=pi[i][j])
                    z = np.random.normal(mu[i][j][idx], sigma[i][j][idx] * self.temperature)
                samples[i][j] = z

        return samples

    def get_initial_state(self):
        return self.lstm_layer.cell.get_initial_state(batch_size=1, dtype=tf.float32)


class _Controller(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        self.out_layer = kl.Dense(config.action_num)

        # 重みを初期化
        dummy_z = np.zeros(shape=(1, config.z_size), dtype=np.float32)
        dummy_h = [
            np.zeros(shape=(1, config.rnn_units), dtype=np.float32),
            np.zeros(shape=(1, config.rnn_units), dtype=np.float32),
        ]
        self(dummy_z, dummy_h)

    def call(self, z, hidden_state):
        x = tf.concat([z, hidden_state[1]], axis=1)
        return self.out_layer(x)

    def get_flat_params(self):
        params = self.get_weights()
        params_list = [tf.reshape(layer, [1, -1]) for layer in params]
        params_list = tf.concat(params_list, axis=1)
        return tf.reshape(params_list, [-1]).numpy()

    def set_flat_params(self, flat_params):
        n = 0
        weights = []
        for layer in self.trainable_variables:
            # shape のサイズを計算(各要素を掛け合わせる)
            size = reduce(lambda a, b: a * b, layer.shape)
            w = tf.reshape(flat_params[n : n + size], layer.shape)
            weights.append(w)
            n += size
        self.set_weights(weights)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.vae = _VAE(self.config)
        self.rnn = _MDNRNN(self.config)
        self.controller = _Controller(self.config)

    def call_restore(self, data: Any, **kwargs) -> None:
        self.vae.set_weights(data[0])
        self.rnn.set_weights(data[1])
        self.controller.set_weights(data[2])

    def call_backup(self, **kwargs) -> Any:
        return [
            self.vae.get_weights(),
            self.rnn.get_weights(),
            self.controller.get_weights(),
        ]

    def summary(self, **kwargs):
        self.vae.summary(**kwargs)
        self.rnn.summary(**kwargs)
        self.controller.summary(**kwargs)


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter
        self.remote_memory: RemoteMemory = self.remote_memory

        self.optimizer = keras.optimizers.Adam(learning_rate=self.config.lr)
        self.q_loss = keras.losses.Huber()

        self.c_score = -np.inf

        self.train_count = 0
        self.sync_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):
        _info = {}

        if self.config.train_mode == 1:
            _info.update(self._train_vae())
            self.train_count += 1
        elif self.config.train_mode == 2:
            _info.update(self._train_rnn())
            self.train_count += 1
        elif self.config.train_mode == 3:
            params, score = self.remote_memory.c_get()
            if params is not None and self.c_score < score:
                self.c_score = score
                self.parameter.controller.set_flat_params(params)

        return _info

    def _train_vae(self):
        if self.remote_memory.vae_length() < self.config.memory_warmup_size:
            return {}
        states = self.remote_memory.vae_sample(self.config.batch_size)
        x = np.asarray(states)

        # --- VAE
        with tf.GradientTape() as tape:
            z_mean, z_log_stddev, z = self.parameter.vae.encode(x, training=True)  # type:ignore , ignore check "None"
            pred_x = self.parameter.vae.decode(z, training=True)

            if self.parameter.vae.use_image_head:
                # reconstruction loss (logistic), commented out.
                """
                eps = 1e-6  # avoid taking log of zero
                rc_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        -(x * tf.math.log(pred_x + eps) + (1.0 - x) * tf.math.log(1.0 - pred_x + eps)),
                        axis=[1, 2, 3],
                    )
                )
                """

                # reconstruction loss (MSE)
                rc_loss = tf.reduce_sum(tf.square(x - pred_x), axis=[1, 2, 3])
                rc_loss = tf.reduce_mean(rc_loss)
            else:
                rc_loss = tf.reduce_sum(tf.square(x - pred_x), axis=1)
                rc_loss = tf.reduce_mean(rc_loss)

            # KL loss
            kl_loss = -0.5 * tf.reduce_sum(
                1 + z_log_stddev - tf.square(z_mean) - tf.exp(z_log_stddev),  # type:ignore , ignore check "None"
                axis=1,
            )
            kl_loss = tf.maximum(kl_loss, self.config.kl_tolerance * self.config.z_size)
            kl_loss = tf.reduce_mean(kl_loss)

            vae_loss = rc_loss + kl_loss
            vae_loss += tf.reduce_sum(self.parameter.vae.losses)  # 正則化項

        grads = tape.gradient(vae_loss, self.parameter.vae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.vae.trainable_variables))

        return {
            "vae_loss": vae_loss.numpy() - (self.config.kl_tolerance * self.config.z_size),
            "rc_loss": rc_loss.numpy(),
            "kl_loss": kl_loss.numpy() - (self.config.kl_tolerance * self.config.z_size),
        }

    def _train_rnn(self):
        if self.remote_memory.rnn_length() < self.config.memory_warmup_size:
            return {}
        batchs = self.remote_memory.rnn_sample(self.config.batch_size)

        states = np.asarray([b["states"] for b in batchs])
        actions = [b["actions"] for b in batchs]
        onehot_actions = tf.one_hot(actions, self.config.action_num, axis=2)

        # encode
        states = states.reshape((self.config.batch_size * (self.config.sequence_length + 1),) + states.shape[2:])
        z = self.parameter.vae.encode(states).numpy()  # type:ignore , ignore check "None"
        z = z.reshape((self.config.batch_size, self.config.sequence_length + 1, -1))

        # --- MDN-RNN
        z1 = z[:, :-1, ...]
        z2 = z[:, 1:, ...]
        z2 = z2.reshape((self.config.batch_size * self.config.sequence_length, -1, 1))
        with tf.GradientTape() as tape:
            pi, mu, log_sigma, _ = self.parameter.rnn(
                z1, onehot_actions, None, return_rnn_only=False, training=True
            )  # type:ignore , ignore check "None"

            # log softmax
            pi = pi - tf.reduce_max(pi, axis=2, keepdims=True)  # overflow_protection
            log_pi = pi - tf.math.log(tf.reduce_sum(tf.exp(pi), axis=2, keepdims=True))

            # log gauss
            log_gauss = -0.5 * (np.log(2 * np.pi) + 2 * log_sigma + (z2 - mu) ** 2 / tf.exp(log_sigma) ** 2)

            # loss
            loss = tf.reduce_sum(tf.exp(log_pi + log_gauss), axis=2, keepdims=True)
            loss = tf.maximum(loss, 1e-6)  # log(0) 回避
            loss = -tf.math.log(loss)  # type:ignore , ignore check "None"
            loss = tf.reduce_mean(loss)
            loss += tf.reduce_sum(self.parameter.rnn.losses)  # 正則化項

        grads = tape.gradient(loss, self.parameter.rnn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.rnn.trainable_variables))

        return {"rnn_loss": loss.numpy()}


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter
        self.remote_memory: RemoteMemory = self.remote_memory

        self.dummy_state = np.full(self.config.observation_shape, self.config.dummy_state_val, dtype=np.float32)
        self.screen = None

        self.sample_collection = self.config.train_mode == 1 or self.config.train_mode == 2

        # ----------- ES(GA)
        if self.config.train_mode == 3:
            best_params = self.parameter.controller.get_flat_params()
            self.param_length = len(best_params)

            # 初期個体
            self.elite_rewards = [[] for _ in range(self.config.num_individual)]
            self.elite_params = [best_params]
            for _ in range(self.config.num_individual - 1):
                p = np.random.randn(self.param_length) * self.config.randn_sigma
                self.elite_params.append(p)
            self.params_idx = 0

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        if self.training and self.sample_collection:
            self.remote_memory.vae_add(state)
            self._recent_states = [state]
            self.recent_actions = []

        self.hidden_state = self.parameter.rnn.get_initial_state()
        self.prev_hidden_state = self.hidden_state

        if self.training and self.config.train_mode == 3:
            self.parameter.controller.set_flat_params(self.elite_params[self.params_idx])
            self.total_reward = 0

        return {}

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        self.invalid_actions = invalid_actions
        self.state = state
        self.z = self.parameter.vae.encode(state[np.newaxis, ...])

        if self.training and self.sample_collection:
            action = cast(int, self.sample_action())
            if len(self.recent_actions) < self.config.sequence_length:
                self.recent_actions.append(action)
        else:
            q = self.parameter.controller(self.z, self.hidden_state)[0].numpy()  # type:ignore , ignore check "None"
            q = np.array([(-np.inf if i in invalid_actions else v) for i, v in enumerate(q)])
            action = int(np.argmax(q))  # 複数はほぼないので無視

            self.prev_hidden_state = self.hidden_state
            self.hidden_state = self.parameter.rnn.forward(self.z, action, self.hidden_state, return_rnn_only=True)

        self.action = action
        return action, {}

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ):
        if not self.training:
            return {}

        if self.sample_collection:
            self.remote_memory.vae_add(self.state)

            if len(self._recent_states) < self.config.sequence_length + 1:
                self._recent_states.append(next_state)

            if done:
                # states : sequence_length + next_state
                # actions: sequence_length
                for _ in range(self.config.sequence_length - len(self.recent_actions)):
                    self._recent_states.append(self.dummy_state)
                    self.recent_actions.append(random.randint(0, self.config.action_num - 1))
                self.remote_memory.rnn_add(
                    {
                        "states": self._recent_states,
                        "actions": self.recent_actions,
                    }
                )

        if self.config.train_mode == 3:
            self.total_reward += reward
            if done:
                self.elite_rewards[self.params_idx].append(self.total_reward)
                if len(self.elite_rewards[self.params_idx]) == self.config.num_simulations:
                    self.params_idx += 1

                # 一通り個体が評価されたら
                if self.params_idx >= len(self.elite_params):
                    self._eval()
                    self.params_idx = 0
                    self.elite_rewards = [[] for _ in range(self.config.num_individual)]

            return {}

        return {}

    def _eval(self):
        elite_rewards = np.array(self.elite_rewards).mean(axis=1)

        # --- エリート戦略
        next_elite_params = []
        best_idx = np.random.choice(np.where(elite_rewards == elite_rewards.max())[0])
        best_params = self.elite_params[best_idx]
        next_elite_params.append(best_params)

        # send parameter
        self.remote_memory.c_update(best_params, elite_rewards[best_idx])

        weights = elite_rewards - elite_rewards.min()
        if weights.sum() == 0:
            weights = np.full(len(elite_rewards), 1 / len(elite_rewards))
        else:
            weights = weights / weights.sum()

        # --- 子の作成
        while len(next_elite_params) < self.config.num_individual:
            # --- 親個体の選択(ルーレット方式、重複あり)
            idx1 = np.argmax(np.random.multinomial(1, weights))
            idx2 = np.argmax(np.random.multinomial(1, weights))

            # --- BLX-α交叉
            c = []
            for i in range(self.param_length):
                if self.elite_params[idx1][i] < self.elite_params[idx2][i]:
                    xmin = self.elite_params[idx1][i]
                    xmax = self.elite_params[idx2][i]
                else:
                    xmin = self.elite_params[idx2][i]
                    xmax = self.elite_params[idx1][i]
                dx = xmax - xmin
                rmin = xmin - self.config.blx_a * dx
                rmax = xmax + self.config.blx_a * dx
                _c = (rmax - rmin) * random.random() + rmin

                # 突然変異
                if random.random() < self.config.mutation:
                    _c = np.random.randn() * self.config.randn_sigma

                c.append(_c)
            next_elite_params.append(c)

        self.elite_params = next_elite_params

    def render_terminal(self, worker, **kwargs) -> None:
        # --- vae
        pred_state = self.parameter.vae.decode(self.z)[0].numpy()  # type:ignore , ignore check "None"
        rmse = np.sqrt(np.mean((self.state - pred_state) ** 2))
        print(f"VAE RMSE: {rmse:.5f}")

    def render_rgb_array(self, worker: WorkerRun, **kwargs) -> Optional[np.ndarray]:
        if self.config.env_observation_type != EnvObservationTypes.COLOR:
            return None

        from srl.utils import pygame_wrapper as pw

        _view_action = 4
        _view_sample = 3
        IMG_W = 64
        IMG_H = 64
        PADDING = 4
        WIDTH = (IMG_W + PADDING) * _view_action + 5
        HEIGHT = (IMG_H + PADDING) * (_view_sample + 1) + 15 * 2 + 5

        if self.screen is None:
            self.screen = pw.create_surface(WIDTH, HEIGHT)
        pw.draw_fill(self.screen, color=(0, 0, 0))

        img1 = self.state * 255
        img2 = self.parameter.vae.decode(self.z)[0].numpy() * 255  # type:ignore , ignore check "None"

        pw.draw_text(self.screen, 0, 0, "original", color=(255, 255, 255))
        pw.draw_image_rgb_array(self.screen, 0, 15, img1)
        pw.draw_text(self.screen, IMG_W + PADDING, 0, "decode", color=(255, 255, 255))
        pw.draw_image_rgb_array(self.screen, IMG_W + PADDING, 15, img2)

        # 横にアクション後の結果を表示
        invalid_actions = worker.get_invalid_actions()
        for i, a in enumerate(range(self.config.action_num)):
            if a in invalid_actions:
                continue
            if i > _view_action:
                break

            pi, mu, log_sigma, _ = self.parameter.rnn.forward(
                self.z, a, self.prev_hidden_state, return_rnn_only=False
            )  # type:ignore , ignore check "None"
            pw.draw_text(
                self.screen,
                (IMG_W + PADDING) * i,
                20 + IMG_H,
                f"action {worker.env.action_to_str(a)}",
                color=(255, 255, 255),
            )

            # 縦にいくつかサンプルを表示
            for j in range(_view_sample):
                n_z = self.parameter.rnn.sample(pi, mu, log_sigma)
                n_img = self.parameter.vae.decode(n_z)[0].numpy() * 255

                x = (IMG_W + PADDING) * i
                y = 20 + IMG_H + 15 + (IMG_H + PADDING) * j
                pw.draw_image_rgb_array(self.screen, x, y, n_img)

        return pw.get_rgb_array(self.screen)
