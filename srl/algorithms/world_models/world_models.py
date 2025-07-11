import collections
import pickle
import random
from functools import reduce
from typing import Any, Optional, cast

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import SpaceTypes
from srl.base.rl.algorithms.base_dqn import RLWorker
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.trainer import RLTrainer
from srl.rl.tf.model import KerasModelAddedSummary
from srl.utils.common import compare_less_version

from .config import Config

kl = keras.layers
v216_older = compare_less_version(tf.__version__, "2.16.0")


class Memory(RLMemory[Config]):
    def setup(self):
        self.vae_buffer = collections.deque(maxlen=self.config.capacity)
        self.rnn_buffer = collections.deque(maxlen=self.config.capacity)

        self.c_score = -np.inf
        self.c_params = None

        self.register_worker_func(self.add_vae, pickle.dumps)
        self.register_worker_func(self.add_rnn, pickle.dumps)
        self.register_worker_func(self.add_c, lambda x1, x2: (x1, x2))
        self.register_trainer_recv_func(self.sample_vae)
        self.register_trainer_recv_func(self.sample_rnn)

    def length(self) -> int:
        if self.config.train_mode == "vae":
            return len(self.vae_buffer)
        elif self.config.train_mode == "rnn":
            return len(self.rnn_buffer)
        return len(self.vae_buffer) + len(self.rnn_buffer)

    def add_vae(self, batch: Any, serialized: bool = False) -> None:
        if serialized:
            batch = pickle.loads(batch)
        self.vae_buffer.append(batch)

    def add_rnn(self, batch: Any, serialized: bool = False) -> None:
        if serialized:
            batch = pickle.loads(batch)
        self.rnn_buffer.append(batch)

    def add_c(self, params, score, serialized: bool = False) -> None:
        if self.c_score < score:
            self.c_score = score
            self.c_params = params

    def sample_vae(self) -> Any:
        if len(self.vae_buffer) < self.config.warmup_size:
            return None
        return random.sample(self.vae_buffer, self.config.batch_size)

    def sample_rnn(self) -> Any:
        if len(self.rnn_buffer) < self.config.warmup_size:
            return None
        return random.sample(self.rnn_buffer, self.config.batch_size)

    def call_restore(self, data: Any, **kwargs) -> None:
        self.vae_buffer = data[0]
        self.rnn_buffer = data[1]
        self.c_score = data[2]
        self.c_params = data[3]

    def call_backup(self, **kwargs):
        return [
            self.vae_buffer,
            self.rnn_buffer,
            self.c_score,
            self.c_params,
        ]

    # ---------------------------

    def c_get(self):
        return self.c_params, self.c_score


class VAE(KerasModelAddedSummary):
    def __init__(self, config: Config):
        super().__init__()

        self.z_size = config.z_size
        self.kl_tolerance = config.kl_tolerance
        self.use_image_head = config.observation_space.is_image()

        # --- encoder
        if self.use_image_head:
            self.encoder_in_layers = [
                kl.Conv2D(filters=32, kernel_size=4, strides=2, activation="relu"),
                kl.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu"),
                kl.Conv2D(filters=128, kernel_size=4, strides=2, activation="relu"),
                kl.Conv2D(filters=256, kernel_size=4, strides=2, activation="relu"),
                kl.Flatten(),
            ]
        else:
            self.encoder_in_layers = [
                kl.Flatten(),
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
            flatten_shape = np.zeros(config.observation_space.shape).flatten().shape
            self.decoder_in_layers = [
                kl.Dense(256, activation="relu"),
                kl.Dense(256, activation="relu"),
                kl.Dense(flatten_shape[0]),
                kl.Reshape(config.observation_space.shape),
            ]

        # build
        self(np.zeros((1,) + config.observation_space.shape))

    def call(self, x, training=False):
        return self.decode(self.encode(x, training=training), training=training)

    def encode(self, x, training=False):
        for layer in self.encoder_in_layers:
            x = layer(x, training=training)
        z_mean = self.encoder_z_mean_layer(x, training=training)
        z_log_stddev = self.encoder_z_log_stddev(x, training=training)

        if x.shape[0] is None:
            return z_mean

        # reparameterize
        e = tf.random.normal(z_mean.shape)
        z = z_mean + tf.exp(0.5 * z_log_stddev) * e

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


class MDNRNN(KerasModelAddedSummary):
    def __init__(self, config: Config):
        super().__init__()

        self.action_num = config.action_space.n
        self.z_size = config.z_size
        self.num_mixture = config.num_mixture
        self.temperature = config.temperature

        # --- LSTM
        self.lstm_layer = kl.LSTM(config.rnn_units, return_sequences=True, return_state=True)

        # --- MDN
        self.mdn_layer = kl.Dense(config.z_size * config.num_mixture * 3)

        # 重みを初期化
        self(
            np.zeros(shape=(1, 1, config.z_size), dtype=config.dtype),
            np.zeros(shape=(1, 1, config.action_space.n), dtype=config.dtype),
            self.get_initial_state(1),
            return_rnn_only=False,
            training=False,
        )

    def call(self, z, onehot_actions, hidden_state, return_rnn_only, training=False):
        batch_size = z.shape[0]
        timesteps = z.shape[1]

        # (batch, timesteps, z + action) -> (batch, timesteps, lstm_dim)
        x = tf.concat([z, onehot_actions], axis=2)
        x, h, c = self.lstm_layer(x, initial_state=hidden_state, training=training)
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

    def get_initial_state(self, batch_size=1):
        if v216_older:
            return self.lstm_layer.cell.get_initial_state(batch_size=batch_size, dtype=self.dtype)
        else:
            return self.lstm_layer.cell.get_initial_state(batch_size)


class Controller(KerasModelAddedSummary):
    def __init__(self, config: Config):
        super().__init__()

        self.out_layer = kl.Dense(config.action_space.n)

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


class Parameter(RLParameter[Config]):
    def setup(self):
        self.vae = VAE(self.config)
        self.rnn = MDNRNN(self.config)
        self.controller = Controller(self.config)

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


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        lr = self.config.lr_scheduler.apply_tf_scheduler(self.config.lr)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.q_loss = keras.losses.Huber()

        self.c_score = -np.inf

        self.sync_count = 0

    def train(self) -> None:
        if self.config.train_mode == "vae":
            self._train_vae()
        elif self.config.train_mode == "rnn":
            self._train_rnn()
            self.train_count += 1
        elif self.config.train_mode == "controller":
            params, score = self.memory.c_get()
            if params is not None and self.c_score < score:
                self.c_score = score
                self.parameter.controller.set_flat_params(params)

    def _train_vae(self):
        batches = self.memory.sample_vae()
        if batches is None:
            return
        self.train_count += 1

        x = np.asarray(batches)

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

        self.info["vae_loss"] = vae_loss.numpy() - (self.config.kl_tolerance * self.config.z_size)
        self.info["rc_loss"] = rc_loss.numpy()
        self.info["kl_loss"] = kl_loss.numpy() - (self.config.kl_tolerance * self.config.z_size)

    def _train_rnn(self):
        batches = self.memory.sample_rnn()
        if batches is None:
            return

        states = np.asarray([b["states"] for b in batches])
        actions = [b["actions"] for b in batches]
        onehot_actions = tf.one_hot(actions, self.config.action_space.n, axis=2)

        # encode
        states = states.reshape((self.config.batch_size * (self.config.sequence_length + 1),) + states.shape[2:])
        z = self.parameter.vae.encode(states).numpy()
        z = z.reshape((self.config.batch_size, self.config.sequence_length + 1, -1))

        # --- MDN-RNN
        z1 = z[:, :-1, ...]
        z2 = z[:, 1:, ...]
        z2 = z2.reshape((self.config.batch_size * self.config.sequence_length, -1, 1))
        with tf.GradientTape() as tape:
            pi, mu, log_sigma, _ = self.parameter.rnn(z1, onehot_actions, None, return_rnn_only=False, training=True)

            # log softmax
            pi = pi - tf.reduce_max(pi, axis=2, keepdims=True)  # overflow_protection
            log_pi = pi - tf.math.log(tf.reduce_sum(tf.exp(pi), axis=2, keepdims=True))

            # log gauss
            log_gauss = -0.5 * (np.log(2 * np.pi) + 2 * log_sigma + (z2 - mu) ** 2 / tf.exp(log_sigma) ** 2)

            # loss
            loss = tf.reduce_sum(tf.exp(log_pi + log_gauss), axis=2, keepdims=True)
            loss = tf.maximum(loss, 1e-6)  # log(0) 回避
            loss = -tf.math.log(loss)
            loss = tf.reduce_mean(loss)
            loss += tf.reduce_sum(self.parameter.rnn.losses)  # 正則化項

        grads = tape.gradient(loss, self.parameter.rnn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.rnn.trainable_variables))

        self.info["rnn_loss"] = loss.numpy()


class Worker(RLWorker[Config, Parameter, Memory]):
    def on_setup(self, worker, context):
        self.screen = None
        self.dummy_state = np.zeros(self.config.observation_space.shape, dtype=np.float32)
        self.sample_collection = self.config.train_mode == "vae" or self.config.train_mode == "rnn"

        # ----------- ES(GA)
        if self.config.train_mode == "controller":
            best_params = self.parameter.controller.get_flat_params()
            self.param_length = len(best_params)

            # 初期個体
            self.elite_rewards = [[] for _ in range(self.config.num_individual)]
            self.elite_params = [best_params]
            for _ in range(self.config.num_individual - 1):
                p = np.random.randn(self.param_length) * self.config.randn_sigma
                self.elite_params.append(p)
            self.params_idx = 0

    def on_reset(self, worker):
        if self.training and self.sample_collection:
            self.memory.add_vae(worker.state)
            self.recent_states = [worker.state]
            self.recent_actions = []

        self.hidden_state = self.parameter.rnn.get_initial_state()
        self.prev_hidden_state = self.hidden_state

        if self.training and self.config.train_mode == "controller":
            self.parameter.controller.set_flat_params(self.elite_params[self.params_idx])
            self.total_reward = 0

    def policy(self, worker) -> Any:
        self.invalid_actions = worker.invalid_actions
        self.state = worker.state
        self.z = self.parameter.vae.encode(worker.state[np.newaxis, ...])

        if self.training and self.sample_collection:
            action = cast(int, self.sample_action())
            if len(self.recent_actions) < self.config.sequence_length:
                self.recent_actions.append(action)
        else:
            q = self.parameter.controller(self.z, self.hidden_state)[0].numpy()  # type:ignore , ignore check "None"
            q = np.array([(-np.inf if i in self.invalid_actions else v) for i, v in enumerate(q)])
            action = int(np.argmax(q))  # 複数はほぼないので無視

            self.prev_hidden_state = self.hidden_state
            self.hidden_state = self.parameter.rnn.forward(self.z, action, self.hidden_state, return_rnn_only=True)

        self.action = action
        return action

    def on_step(self, worker):
        if not self.training:
            return

        if self.sample_collection:
            self.memory.add_vae(self.state)

            if len(self.recent_states) < self.config.sequence_length + 1:
                self.recent_states.append(worker.next_state)

            if worker.done:
                # states : sequence_length + next_state
                # actions: sequence_length
                for _ in range(self.config.sequence_length - len(self.recent_actions)):
                    self.recent_states.append(self.dummy_state)
                    self.recent_actions.append(random.randint(0, self.config.action_space.n - 1))
                self.memory.add_rnn(
                    {
                        "states": self.recent_states,
                        "actions": self.recent_actions,
                    },
                )

        if self.config.train_mode == "controller":
            self.total_reward += worker.reward
            if worker.done:
                self.elite_rewards[self.params_idx].append(self.total_reward)
                if len(self.elite_rewards[self.params_idx]) == self.config.num_simulations:
                    self.params_idx += 1

                # 一通り個体が評価されたら
                if self.params_idx >= len(self.elite_params):
                    self._eval()
                    self.params_idx = 0
                    self.elite_rewards = [[] for _ in range(self.config.num_individual)]

    def _eval(self):
        elite_rewards = np.array(self.elite_rewards).mean(axis=1)

        # --- エリート戦略
        next_elite_params = []
        best_idx = np.random.choice(np.where(elite_rewards == elite_rewards.max())[0])
        best_params = self.elite_params[best_idx]
        next_elite_params.append(best_params)

        # send parameter
        self.memory.add_c(best_params, elite_rewards[best_idx])

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

    def render_rgb_array(self, worker, **kwargs) -> Optional[np.ndarray]:
        if self.config.observation_space.stype != SpaceTypes.COLOR:
            return None

        from srl.utils import pygame_wrapper as pw

        _view_action = 4
        _view_sample = 3
        IMG_W = 64
        IMG_H = 64
        STR_H = 20
        PADDING = 4
        WIDTH = (IMG_W + PADDING) * _view_action + 10
        HEIGHT = (IMG_H + PADDING) * (_view_sample + 1) + STR_H * 2 + 10

        if self.screen is None:
            self.screen = pw.create_surface(WIDTH, HEIGHT)
        pw.draw_fill(self.screen, color=(0, 0, 0))

        img1 = self.state * 255
        img2 = self.parameter.vae.decode(self.z)[0].numpy() * 255  # type:ignore , ignore check "None"

        pw.draw_text(self.screen, 0, 0, "origin", color=(255, 255, 255))
        pw.draw_image_rgb_array(self.screen, 0, STR_H, img1)
        pw.draw_text(self.screen, IMG_W + PADDING, 0, "decode", color=(255, 255, 255))
        pw.draw_image_rgb_array(self.screen, IMG_W + PADDING, STR_H, img2)

        # 横にアクション後の結果を表示
        for i, a in enumerate(range(self.config.action_space.n)):
            if a in worker.invalid_actions:
                continue
            if i > _view_action:
                break

            pi, mu, log_sigma, _ = self.parameter.rnn.forward(self.z, a, self.prev_hidden_state, return_rnn_only=False)
            pw.draw_text(
                self.screen,
                (IMG_W + PADDING) * i,
                20 + IMG_H,
                f"act {worker.env.action_to_str(a)}",
                color=(255, 255, 255),
            )

            # 縦にいくつかサンプルを表示
            for j in range(_view_sample):
                n_z = self.parameter.rnn.sample(pi, mu, log_sigma)
                n_img = self.parameter.vae.decode(n_z)[0].numpy() * 255

                x = (IMG_W + PADDING) * i
                y = 20 + IMG_H + STR_H + (IMG_H + PADDING) * j
                pw.draw_image_rgb_array(self.screen, x, y, n_img)

        return pw.get_rgb_array(self.screen)
