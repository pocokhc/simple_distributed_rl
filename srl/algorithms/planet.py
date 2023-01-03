import random
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, cast

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
from srl.base.rl.remote_memory import ExperienceReplayBuffer
from srl.rl.functions.common_tf import gaussian_kl_divergence
from srl.rl.models.tf.input_layer import create_input_layer
from srl.rl.models.tf.mlp_block import MLPBlock

"""
paper: https://arxiv.org/abs/1811.04551
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    lr: float = 0.001
    batch_size: int = 32
    capacity: int = 100_000
    memory_warmup_size: int = 1000

    enable_overshooting_loss: bool = False

    # model
    z_size: int = 64
    sequence_length: int = 100
    rnn_units: int = 256
    hidden_block: kl.Layer = MLPBlock
    hidden_block_kwargs: dict = field(default_factory=dict)
    vae_beta: float = 1.5
    num_overshooting: int = 2
    reward_block: kl.Layer = MLPBlock
    reward_block_kwargs: dict = field(default_factory=dict)

    # GA
    pred_action_length: int = 5
    num_generation: int = 10
    num_individual: int = 5
    num_simulations: int = 20
    mutation: float = 0.1
    print_ga_debug: bool = True

    # other
    dummy_state_val: float = 0.0

    def set_processor(self) -> List[Processor]:
        return [
            ImageProcessor(
                image_type=EnvObservationType.COLOR,
                resize=(64, 64),
                enable_norm=True,
            )
        ]

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.CONTINUOUS

    @staticmethod
    def getName() -> str:
        return "PlaNet"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.memory_warmup_size < self.capacity
        assert self.batch_size < self.memory_warmup_size
        assert self.num_overshooting > 0


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
class RemoteMemory(ExperienceReplayBuffer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.init(self.config.capacity)


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _DynamicsModel(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        self.z_size = config.z_size
        self.action_num = config.action_num

        # --- encoder
        in_layer, c, use_image_head = create_input_layer(
            config.observation_shape,
            config.env_observation_type,
        )
        assert use_image_head, "Input supports only image format."
        assert config.window_length == 1
        c = kl.Conv2D(filters=32, kernel_size=4, strides=2, activation="relu")(c)
        c = kl.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")(c)
        c = kl.Conv2D(filters=128, kernel_size=4, strides=2, activation="relu")(c)
        c = kl.Conv2D(filters=256, kernel_size=4, strides=2, activation="relu")(c)
        c = kl.Flatten()(c)
        z_mean = kl.Dense(config.z_size)(c)
        z_log_stddev = kl.Dense(config.z_size)(c)
        self.encoder = keras.Model(in_layer, [z_mean, z_log_stddev], name="Encoder")

        # --- decoder(Observation model)
        in_layer = c = kl.Input(shape=(config.z_size + config.rnn_units,))
        c = kl.Dense(2 * 2 * 256, activation="relu")(c)
        c = kl.Reshape((1, 1, 2 * 2 * 256))(c)
        c = kl.Conv2DTranspose(128, kernel_size=5, strides=2, padding="valid", activation="relu")(c)
        c = kl.Conv2DTranspose(64, kernel_size=5, strides=2, padding="valid", activation="relu")(c)
        c = kl.Conv2DTranspose(32, kernel_size=6, strides=2, padding="valid", activation="relu")(c)
        c = kl.Conv2DTranspose(3, kernel_size=6, strides=2, padding="valid", activation="sigmoid")(c)
        self.decoder = keras.Model(in_layer, c, name="ObservationModel")

        # --- RNN(Deterministic state model)
        self.rnn_layer = kl.GRU(
            config.rnn_units, return_sequences=True, return_state=True, name="DeterministicStateModel"
        )

        # ---  Gaussian(Stochastic state model)
        in_layer = c = kl.Input(shape=(config.rnn_units,))
        c = config.hidden_block(**config.hidden_block_kwargs)(c)
        c1 = kl.Dense(config.z_size)(c)
        c2 = kl.Dense(config.z_size)(c)
        self.stochastic_model = keras.Model(in_layer, [c1, c2], name="StochasticStateModel")

        # 重みを初期化
        dummy_state = np.zeros(shape=(1,) + config.observation_shape, dtype=np.float32)
        dummy_action = 0
        self(dummy_state, dummy_action, None)

    # 初期化用
    def call(self, state, action, hidden_state):
        z = self.encode(state)
        x, _ = self.one_step_transition(z, action, hidden_state)
        z_mean, z_log_stddev = self.stochastic_model(x)
        _ = tf.concat([z, x], axis=1)
        return z_mean, z_log_stddev

    def encode(self, x, training=False):
        z_mean, z_log_stddev = self.encoder(x, training=training)

        # reparameterize
        e = tf.random.normal(z_mean.shape)
        z = z_mean + tf.exp(z_log_stddev) * e

        if training:
            return z_mean, z_log_stddev, z
        else:
            return z

    def decode(self, latent_space, training=False):
        return self.decoder(latent_space, training=training)

    def pred_z(self, h, training=False):
        z_mean, z_log_stddev = self.stochastic_model(h, training=training)

        # reparameterize
        e = tf.random.normal(z_mean.shape)
        z = z_mean + tf.exp(z_log_stddev) * e

        if training:
            return z_mean, z_log_stddev, z
        else:
            return z

    def one_step_transition(self, z, action: int, hidden_state):
        onehot_action = tf.one_hot(action, self.action_num)

        # (batch, seq, shape)
        z = tf.reshape(z, (1, 1, -1))
        onehot_action = tf.reshape(onehot_action, (1, 1, -1))

        x = tf.concat([z, onehot_action], axis=2)
        x, hidden_state = self.rnn_layer(x, initial_state=hidden_state, training=False)

        # -> (batch, shape)
        x = tf.reshape(x, (1, -1))

        return x, hidden_state

    def get_initial_state(self):
        return self.rnn_layer.cell.get_initial_state(batch_size=1, dtype=tf.float32)


class _RewardModel(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        in_layer = c = kl.Input(shape=(config.z_size + config.rnn_units,))
        c = config.reward_block(**config.reward_block_kwargs)(c)
        c1 = kl.Dense(1)(c)
        c2 = kl.Dense(1)(c)
        self.reward_model = keras.Model(in_layer, [c1, c2], name="RewardModel")

        # 重みを初期化
        dummy_state = np.zeros(shape=(1, config.z_size + config.rnn_units), dtype=np.float32)
        self(dummy_state)

    def call(self, latent_space, training=False):
        r_mean, r_log_stddev = self.reward_model(latent_space, training=training)

        # reparameterize
        e = tf.random.normal(r_mean.shape)
        r = r_mean + tf.exp(r_log_stddev) * e

        return r


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.model = _DynamicsModel(self.config)
        self.reward_model = _RewardModel(self.config)

    def call_restore(self, data: Any, **kwargs) -> None:
        self.model.set_weights(data[0])
        self.reward_model.set_weights(data[1])

    def call_backup(self, **kwargs) -> Any:
        return [
            self.model.get_weights(),
            self.reward_model.get_weights(),
        ]

    def summary(self, **kwargs):
        self.model.summary(**kwargs)
        self.reward_model.summary(**kwargs)


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        self.optimizer = keras.optimizers.Adam(learning_rate=self.config.lr)
        self.train_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):
        if self.config.enable_overshooting_loss:
            return self._train_latent_overshooting_loss()
        else:
            return self._train()

    def _train(self):
        if self.remote_memory.length() < self.config.memory_warmup_size:
            return {}
        batchs = self.remote_memory.sample(self.config.batch_size)

        # state
        states = np.asarray([b["states"] for b in batchs])
        states = states.reshape(
            (self.config.batch_size * (self.config.sequence_length + 1),) + states.shape[2:]
        )  # (batch * seq, shape)

        # action
        actions = [b["actions"] for b in batchs]
        onehot_actions = tf.one_hot(actions, self.config.action_num, axis=2)

        # reward
        rewards = np.asarray([[0] + b["rewards"] for b in batchs])
        rewards = rewards.reshape((self.config.batch_size * (self.config.sequence_length + 1), -1))

        model = self.parameter.model
        with tf.GradientTape() as tape:
            z_mean, z_log_stddev, z = model.encode(states, training=True)
            # (batch * seq, shape) -> (batch, seq, shape)
            seq_z = tf.reshape(z, (self.config.batch_size, self.config.sequence_length + 1, -1))
            seq_z1 = seq_z[:, :-1, ...]  # prev

            # deterministic
            h = tf.concat([seq_z1, onehot_actions], axis=2)
            h, _ = model.rnn_layer(h, initial_state=None, training=True)
            h0 = np.zeros((h.shape[0], 1, h.shape[2]))
            h = tf.concat([h0, h], axis=1)
            h = tf.reshape(h, (self.config.batch_size * (self.config.sequence_length + 1), -1))

            latent_space = tf.concat([z, h], axis=1)

            # --- reconstruction loss (binary_crossentropy)
            pred_states = model.decode(latent_space, training=True)
            eps = 1e-6  # avoid taking log of zero
            rc_loss = tf.reduce_mean(
                -(states * tf.math.log(pred_states + eps) + (1.0 - states) * tf.math.log(1.0 - pred_states + eps)),
                axis=[1, 2, 3],
            )

            # --- kl_loss, stochastic
            pred_z_mean, pred_z_log_stddev, _ = model.pred_z(h, training=True)
            kl_loss = gaussian_kl_divergence(z_mean, z_log_stddev, pred_z_mean, pred_z_log_stddev)
            kl_loss = self.config.vae_beta * tf.reduce_mean(kl_loss, axis=1)

            loss = tf.reduce_mean(rc_loss + kl_loss)
            loss += tf.reduce_sum(model.losses)  # 正則化項

        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # --- reward
        with tf.GradientTape() as tape:
            pred_reward = self.parameter.reward_model(latent_space, training=True)
            reward_loss = tf.reduce_mean(tf.square(rewards - pred_reward), axis=1)
            reward_loss += tf.reduce_sum(self.parameter.reward_model.losses)  # 正則化項

        grads = tape.gradient(reward_loss, self.parameter.reward_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.reward_model.trainable_variables))

        self.train_count += 1
        return {
            "loss": loss.numpy(),
            "rc_loss": np.mean(rc_loss),
            "kl_loss": np.mean(kl_loss),
            "reward_loss": np.mean(reward_loss),
        }

    def _train_latent_overshooting_loss(self):
        if self.remote_memory.length() < self.config.memory_warmup_size:
            return {}
        batchs = self.remote_memory.sample(self.config.batch_size)

        # state
        states = np.asarray([b["states"] for b in batchs])
        flat_states = states.reshape(
            (self.config.batch_size * (self.config.sequence_length + 1),) + states.shape[2:]
        )  # (batch * seq, shape)

        # action
        actions = [b["actions"] for b in batchs]
        onehot_actions = tf.one_hot(actions, self.config.action_num, axis=2)

        # reward
        rewards = np.asarray([b["rewards"] for b in batchs])
        rewards = rewards.reshape((self.config.batch_size * self.config.sequence_length, -1))

        model = self.parameter.model
        hidden_state = model.get_initial_state()
        hidden_state = tf.tile(hidden_state, [self.config.batch_size, 1])
        latent_space_list = []
        eps = 1e-6  # avoid taking log of zero
        with tf.GradientTape() as tape:
            z_mean, z_log_stddev, z = model.encode(flat_states, training=True)
            # (batch * seq, shape) -> (batch, seq, shape)
            z = tf.reshape(z, (self.config.batch_size, self.config.sequence_length + 1, -1))
            z_mean = tf.reshape(z_mean, (self.config.batch_size, self.config.sequence_length + 1, -1))
            z_log_stddev = tf.reshape(z_log_stddev, (self.config.batch_size, self.config.sequence_length + 1, -1))

            rc_loss = tf.constant([0] * self.config.batch_size, dtype=tf.float32)
            kl_loss = tf.constant([0] * self.config.batch_size, dtype=tf.float32)
            overshooting_list = []
            for step in range(self.config.sequence_length):
                step_z = z[:, step, ...]
                step_s = tf.concat([step_z, hidden_state], axis=1)
                latent_space_list.append(step_s)

                # --- reconstruction loss
                step_state = states[:, step, ...]
                pred_state = model.decode(step_s, training=True)
                rc_loss += tf.reduce_mean(
                    -(
                        step_state * tf.math.log(pred_state + eps)
                        + (1.0 - step_state) * tf.math.log(1.0 - pred_state + eps)
                    ),
                    axis=[1, 2, 3],
                )

                # --- kl loss
                step_kl_loss = tf.constant([0] * self.config.batch_size, dtype=tf.float32)
                for o_z, o_hidden_state in overshooting_list:
                    # stochastic
                    pred_z_mean, pred_z_log_stddev, _ = model.pred_z(o_hidden_state, training=True)
                    _kl_loss = gaussian_kl_divergence(
                        z_mean[:, step, ...],
                        z_log_stddev[:, step, ...],
                        pred_z_mean,
                        pred_z_log_stddev,
                    )
                    _kl_loss = tf.reduce_mean(_kl_loss, axis=1)
                    step_kl_loss += self.config.vae_beta * _kl_loss
                if len(overshooting_list) > 0:
                    kl_loss += step_kl_loss / len(overshooting_list)

                # --- trans(one step)
                next_overshooting_list = []
                onehot_action = onehot_actions[:, step : step + 1, ...]
                _x = tf.concat([z[:, step : step + 1, ...], onehot_action], axis=2)
                _, hidden_state = model.rnn_layer(_x, initial_state=hidden_state, training=True)
                _, _, o_z = model.pred_z(hidden_state, training=True)
                if len(overshooting_list) < self.config.num_overshooting:
                    next_overshooting_list.append([tf.expand_dims(o_z, axis=1), hidden_state])

                # --- trans(overshooting)
                for o_z, o_hidden_state in overshooting_list:
                    _x = tf.concat([o_z, onehot_action], axis=2)
                    _, o_hidden_state = model.rnn_layer(_x, initial_state=o_hidden_state, training=True)
                    _, _, o_z = model.pred_z(o_hidden_state, training=True)
                    next_overshooting_list.append([tf.expand_dims(o_z, axis=1), o_hidden_state])
                overshooting_list = next_overshooting_list

            rc_loss = rc_loss / self.config.sequence_length
            kl_loss = kl_loss / self.config.sequence_length

            loss = tf.reduce_mean(rc_loss + kl_loss)
            loss += tf.reduce_sum(model.losses)  # 正則化項

        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # --- reward
        latent_space = tf.stack(latent_space_list)
        latent_space = tf.transpose(latent_space, [1, 0, 2])
        latent_space = tf.reshape(latent_space, (latent_space.shape[0] * latent_space.shape[1], latent_space.shape[2]))
        with tf.GradientTape() as tape:
            pred_reward = self.parameter.reward_model(latent_space, training=True)
            reward_loss = tf.reduce_mean(tf.square(rewards - pred_reward), axis=1)
            reward_loss += tf.reduce_sum(self.parameter.reward_model.losses)  # 正則化項

        grads = tape.gradient(reward_loss, self.parameter.reward_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.reward_model.trainable_variables))

        self.train_count += 1
        return {
            "rc_loss": np.mean(rc_loss),
            "kl_loss": np.mean(kl_loss),
            "reward_loss": np.mean(reward_loss),
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

        self.dummy_state = np.full(self.config.observation_shape, self.config.dummy_state_val, dtype=np.float32)
        self.screen = None

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        self._recent_states = [state]
        self.recent_actions = []
        self.recent_rewards = []

        self.hidden_state = self.parameter.model.get_initial_state()
        self.prev_hidden_state = self.hidden_state

        return {}

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        self.invalid_actions = invalid_actions
        self.state = state

        if self.training:
            self.action = self.sample_action()
            self.z = None
        else:
            self.z = self.parameter.model.encode(state[np.newaxis, ...])
            self.action = self._ga_policy(self.z)

            # forward
            self.prev_hidden_state = self.hidden_state
            _, self.hidden_state = self.parameter.model.one_step_transition(self.z, self.action, self.hidden_state)

        if len(self.recent_actions) < self.config.sequence_length:
            self.recent_actions.append(self.action)
        return self.action, {}

    def _ga_policy(self, z):
        # --- 初期個体
        elite_actions = [
            [random.randint(0, self.config.action_num - 1) for a in range(self.config.pred_action_length)]
            for _ in range(self.config.num_individual)
        ]
        best_actions = None

        # --- 世代ループ
        for g in range(self.config.num_generation):
            # --- 個体を評価
            t0 = time.time()
            elite_rewards = []
            for i in range(len(elite_actions)):
                rewards = []
                for _ in range(self.config.num_simulations):
                    reward = self._eval_actions(z, elite_actions[i], self.hidden_state)
                    rewards.append(reward)
                elite_rewards.append(np.mean(rewards))
            elite_rewards = np.array(elite_rewards)

            # --- エリート戦略
            next_elite_actions = []
            best_idx = random.choice(np.where(elite_rewards == elite_rewards.max())[0])
            best_actions = elite_actions[best_idx]
            next_elite_actions.append(best_actions)

            # debug
            if self.config.print_ga_debug:
                print(f"--- {g}/{self.config.num_generation} {time.time()-t0:.1f}s")
                print(f"*{best_idx} {elite_rewards[best_idx]:.5f} {elite_actions[best_idx]}")
                for idx in range(len(elite_actions)):
                    if idx >= 4:
                        print("...")
                        break
                    print(f" {idx} {elite_rewards[idx]:.3f} {elite_actions[idx]}")

            # 最後は交叉しない
            if self.config.num_generation - 1 == g:
                break

            # weight
            weights = elite_rewards - elite_rewards.min()
            if weights.sum() == 0:
                weights = np.full(len(elite_rewards), 1 / len(elite_rewards))
            else:
                weights = weights / weights.sum()

            # --- 子の作成
            while len(next_elite_actions) < self.config.num_individual:
                # --- 親個体の選択(ルーレット方式、重複あり)
                idx1 = np.argmax(np.random.multinomial(1, weights))
                idx2 = np.argmax(np.random.multinomial(1, weights))

                # --- 一様交叉
                c1 = []
                c2 = []
                for i in range(self.config.pred_action_length):
                    if random.random() < 0.5:
                        _c1 = elite_actions[idx1][i]
                        _c2 = elite_actions[idx2][i]
                    else:
                        _c1 = elite_actions[idx2][i]
                        _c2 = elite_actions[idx1][i]

                    # 突然変異
                    if random.random() < self.config.mutation:
                        _c1 = random.randint(0, self.config.action_num - 1)
                    if random.random() < self.config.mutation:
                        _c2 = random.randint(0, self.config.action_num - 1)

                    c1.append(_c1)
                    c2.append(_c2)

                next_elite_actions.append(c1)
                next_elite_actions.append(c2)
            elite_actions = next_elite_actions

        # 一番いい結果のアクションを実行
        return best_actions[0]

    def _eval_actions(self, z, action_list, hidden_state):

        reward = 0
        for step in range(len(action_list)):
            h, hidden_state = self.parameter.model.one_step_transition(z, action_list[step], hidden_state)

            # stochastic
            z = self.parameter.model.pred_z(h)

            # reward
            s = tf.concat([z, h], axis=1)
            pred_reward = self.parameter.reward_model(s)
            reward += pred_reward.numpy()[0][0]

        return reward

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ):
        if not self.training:
            return {}

        if len(self._recent_states) < self.config.sequence_length + 1:
            self._recent_states.append(next_state)
            self.recent_rewards.append(reward)

        if done:
            # states : sequence_length + next_state
            # actions: sequence_length
            for _ in range(self.config.sequence_length - len(self.recent_actions)):
                self._recent_states.append(self.dummy_state)
                self.recent_actions.append(random.randint(0, self.config.action_num - 1))
                self.recent_rewards.append(0)
            self.remote_memory.add(
                {
                    "states": self._recent_states,
                    "actions": self.recent_actions,
                    "rewards": self.recent_rewards,
                }
            )

        return {}

    def render_terminal(self, env, worker, **kwargs) -> None:
        pass

    def render_rgb_array(self, env, worker, **kwargs) -> Optional[np.ndarray]:
        if self.config.env_observation_type != EnvObservationType.COLOR:
            return None
        from srl.utils import pygame_wrapper as pw

        _view_action = 4
        _view_sample = 3
        IMG_W = 64
        IMG_H = 64
        STR_H = 15
        PADDING = 4
        WIDTH = (IMG_W + PADDING) * _view_action + 5
        HEIGHT = (IMG_H + PADDING + STR_H) * (_view_sample + 1) + STR_H * 2 + 5

        if self.screen is None:
            self.screen = pw.create_surface(WIDTH, HEIGHT)
        pw.draw_fill(self.screen, color=(0, 0, 0))

        # --- vae
        s = tf.concat([self.z, self.prev_hidden_state], axis=1)
        pred_state = self.parameter.model.decoder(s)[0].numpy()
        rmse = np.sqrt(np.mean((self.state - pred_state) ** 2))

        img1 = self.state * 255
        img2 = pred_state * 255

        pw.draw_text(self.screen, 0, 0, "original", color=(255, 255, 255))
        pw.draw_image_rgb_array(self.screen, 0, STR_H, img1)
        pw.draw_text(self.screen, IMG_W + PADDING, 0, f"decode(RMSE: {rmse:.5f})", color=(255, 255, 255))
        pw.draw_image_rgb_array(self.screen, IMG_W + PADDING, STR_H, img2)

        # 横にアクション後の結果を表示
        for i, a in enumerate(self.get_valid_actions()):
            if i > _view_action:
                break

            h, _ = self.parameter.model.one_step_transition(self.z, a, self.prev_hidden_state)

            pw.draw_text(
                self.screen, (IMG_W + PADDING) * i, 20 + IMG_H, f"action {env.action_to_str(a)}", color=(255, 255, 255)
            )

            # 縦にいくつかサンプルを表示
            for j in range(_view_sample):
                n_z = self.parameter.model.pred_z(h)

                s = tf.concat([n_z, h], axis=1)
                next_state = self.parameter.model.decode(s)
                reward = self.parameter.reward_model(s)

                n_img = next_state[0].numpy() * 255
                reward = reward.numpy()[0][0]

                x = (IMG_W + PADDING) * i
                y = 20 + IMG_H + STR_H + (IMG_H + PADDING + STR_H) * j
                pw.draw_text(self.screen, x, y, f"{reward:.3f}", color=(255, 255, 255))
                pw.draw_image_rgb_array(self.screen, x, y + STR_H, n_img)

        return pw.get_rgb_array(self.screen)
