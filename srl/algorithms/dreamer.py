import random
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kl
from srl.base.define import EnvObservationType, RLObservationType
from srl.base.rl.algorithms.continuous_action import ContinuousActionConfig, ContinuousActionWorker
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLRemoteMemory, RLTrainer
from srl.base.rl.processor import Processor
from srl.base.rl.processors.image_processor import ImageProcessor
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import ExperienceReplayBuffer
from srl.rl.functions.common_tf import gaussian_kl_divergence
from srl.rl.models.input_layer import create_input_layer
from srl.rl.models.mlp_block import MLPBlock
from srl.utils.viewer import Viewer

"""
paper: https://arxiv.org/abs/1912.01603
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
    hidden_block_kwargs: dict = None
    vae_beta: float = 1.5
    num_overshooting: int = 2

    # GA
    pred_action_length: int = 5
    num_generation: int = 10
    num_individual: int = 5
    num_simulations: int = 20
    mutation: float = 0.1
    print_ga_debug: bool = True

    # other
    dummy_state_val: float = 0.0

    def __post_init__(self):
        super().__init__()

        if self.hidden_block_kwargs is None:
            self.hidden_block_kwargs = dict(
                hidden_layer_sizes=(300, 300),
                activation="elu",
            )

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
        return "Dreamer"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.memory_warmup_size < self.capacity
        assert self.batch_size < self.memory_warmup_size


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
class RemoteMemory(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.dynamics_buffer = ExperienceReplayBuffer(self.config)
        self.behavior_buffer = ExperienceReplayBuffer(self.config)
        self.dynamics_buffer.init(self.config.capacity)
        self.behavior_buffer.init(self.config.capacity)

    def length(self) -> int:
        return self.dynamics_buffer.length() + self.behavior_buffer.length()

    def call_restore(self, data: Any, **kwargs) -> None:
        self.dynamics_buffer.call_restore(data[0], **kwargs)
        self.behavior_buffer.call_restore(data[1], **kwargs)

    def call_backup(self, **kwargs):
        return [
            self.dynamics_buffer.call_backup(**kwargs),
            self.behavior_buffer.call_backup(**kwargs),
        ]

    # ---------------------------
    def dynamics_length(self) -> int:
        return self.dynamics_buffer.length()

    def dynamics_add(self, batch: Any):
        self.dynamics_buffer.add(batch)

    def dynamics_sample(self, batch_size: int) -> List[Any]:
        return self.dynamics_buffer.sample(batch_size)

    def behavior_length(self) -> int:
        return self.behavior_buffer.length()

    def behavior_add(self, batch: Any):
        self.behavior_buffer.add(batch)

    def behavior_sample(self, batch_size: int) -> List[Any]:
        return self.behavior_buffer.sample(batch_size)


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

        # --- reward model
        in_layer = c = kl.Input(shape=(config.z_size + config.rnn_units,))
        c = config.hidden_block(**config.hidden_block_kwargs)(c)
        c1 = kl.Dense(1)(c)
        c2 = kl.Dense(1)(c)
        self.reward_model = keras.Model(in_layer, [c1, c2], name="RewardModel")

        # 重みを初期化
        dummy_state = np.zeros(shape=(1,) + config.observation_shape, dtype=np.float32)
        dummy_action = 0
        self(dummy_state, dummy_action, None)

    # 初期化用
    def call(self, state, action, hidden_state):
        z = self.encode(state)
        x, _ = self.one_step_transition(z, action, hidden_state)
        z_mean, z_log_stddev = self.stochastic_model(x)
        s = tf.concat([z, x], axis=1)
        _ = self.pred_reward(s)
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

    def pred_reward(self, latent_space, training=False):
        r_mean, r_log_stddev = self.reward_model(latent_space, training=training)

        # reparameterize
        e = tf.random.normal(r_mean.shape)
        r = r_mean + tf.exp(r_log_stddev) * e

        return r

    def forward_hidden_state(self, z, action: int, hidden_state):
        onehot_action = tf.one_hot(action, self.action_num)

        # (batch, seq, shape)
        z = tf.reshape(z, (1, 1, -1))
        onehot_action = tf.reshape(onehot_action, (1, 1, -1))

        x = tf.concat([z, onehot_action], axis=2)
        _, hidden_state = self.rnn_layer(x, initial_state=hidden_state, training=False)

        return hidden_state

    def get_initial_state(self):
        return self.rnn_layer.cell.get_initial_state(batch_size=1, dtype=tf.float32)


class _ActionModel(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        self.z_size = config.z_size
        self.action_num = config.action_num

        in_layer = c = kl.Input(shape=(config.z_size + config.rnn_units,))
        c = config.hidden_block(**config.hidden_block_kwargs)(c)
        c1 = kl.Dense(1)(c)
        c2 = kl.Dense(1)(c)
        self.reward_model = keras.Model(in_layer, [c1, c2], name="RewardModel")

        # 重みを初期化
        dummy_x = np.zeros(shape=(1, config.z_size + config.rnn_units), dtype=np.float32)
        self(dummy_x)

    # 初期化用
    def call(self, latent_space, training=False):
        mean, log_stddev = self.reward_model(latent_space, training=training)

        # reparameterize
        e = tf.random.normal(mean.shape)
        action = mean + tf.exp(log_stddev) * e

        # Squashed Gaussian Policy
        action = tf.tanh(action)

        return action


class _ValueModel(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        in_layer = c = kl.Input(shape=(config.z_size + config.rnn_units,))
        c = config.hidden_block(**config.hidden_block_kwargs)(c)
        c = kl.Dense(1)(c)
        self.value_model = keras.Model(in_layer, c, name="ValueModel")

        # 重みを初期化
        dummy_x = np.zeros(shape=(1, config.z_size + config.rnn_units), dtype=np.float32)
        self(dummy_x)

    # 初期化用
    def call(self, latent_space, training=False):
        return self.value_model(latent_space, training=training)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.dynamics_model = _DynamicsModel(self.config)
        self.action_model = _ActionModel(self.config)
        self.value_model = _ValueModel(self.config)

    def call_restore(self, data: Any, **kwargs) -> None:
        self.dynamics_model.set_weights(data[0])
        self.action_model.set_weights(data[1])
        self.value_model.set_weights(data[2])

    def call_backup(self, **kwargs) -> Any:
        return [
            self.dynamics_model.get_weights(),
            self.action_model.get_weights(),
            self.value_model.get_weights(),
        ]

    def summary(self, **kwargs):
        self.dynamics_model.summary(**kwargs)
        self.action_model.summary(**kwargs)
        self.value_model.summary(**kwargs)


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
        _info = {}
        _info.update(self._dynamics_train())
        _info.update(self._behavior_train())
        self.train_count += 1
        return _info

    def _dynamics_train(self):
        if self.remote_memory.dynamics_length() < self.config.memory_warmup_size:
            return {}
        batchs = self.remote_memory.dynamics_sample(self.config.batch_size)

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

        # ------------------------
        # dynamics
        # ------------------------
        model = self.parameter.dynamics_model
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

            # --- reward
            pred_reward = self.parameter.reward_model(latent_space, training=True)
            reward_loss = tf.reduce_mean(tf.square(rewards - pred_reward), axis=1)
            reward_loss += tf.reduce_sum(self.parameter.reward_model.losses)  # 正則化項

            loss = tf.reduce_mean(rc_loss + kl_loss + reward_loss)
            loss += tf.reduce_sum(model.losses)  # 正則化項

        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # --------------------------------
        # behavior
        # --------------------------------
        states = np.asarray([b["states"][0] for b in batchs])
        states = states[:, np.newaxis, ...]

        # value
        with tf.GradientTape() as tape:
            # H step予測する

            pred_reward = self.parameter.reward_model(latent_space, training=True)
            reward_loss = tf.reduce_mean(tf.square(rewards - pred_reward), axis=1)
            reward_loss += tf.reduce_sum(self.parameter.reward_model.losses)  # 正則化項

        grads = tape.gradient(reward_loss, self.parameter.reward_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.reward_model.trainable_variables))

        return {
            "loss": loss.numpy(),
            "rc_loss": np.mean(rc_loss),
            "kl_loss": np.mean(kl_loss),
            "reward_loss": np.mean(reward_loss),
        }

    def _behavior_train(self):
        if self.remote_memory.behavior_length() < self.config.memory_warmup_size:
            return {}
        batchs = self.remote_memory.behavior_sample(self.config.batch_size)

        states = np.asarray([b for b in batchs])
        states = states[:, np.newaxis, ...]

        return {
            "loss": loss.numpy(),
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
        self.viewer = None

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        self.recent_states = [state]
        self.recent_actions = []
        self.recent_rewards = []

        self.hidden_state = self.parameter.dynamics_model.get_initial_state()
        self.prev_hidden_state = self.hidden_state

        return {}

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        self.state = state

        if self.training:
            action = self.sample_action()
            self.latent_space = None
        else:
            z = self.parameter.dynamics_model.encode(state[np.newaxis, ...])
            self.latent_space = tf.concat([z, self.hidden_state], axis=1)

            # action
            action = self.parameter.action_model(self.latent_space)

            # hidden_state
            self.prev_hidden_state = self.hidden_state
            self.hidden_state = self.parameter.dynamics_model.forward_hidden_state(z, action, self.hidden_state)

        # recent
        if len(self.recent_actions) < self.config.sequence_length:
            self.recent_actions.append(action)

        return action, {}

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ) -> dict:
        if not self.training:
            return {}

        self.remote_memory.behavior_add(self.state)

        if len(self.recent_states) < self.config.sequence_length + 1:
            self.recent_states.append(next_state)
            self.recent_rewards.append(reward)

        if done:
            # states : sequence_length + next_state
            # actions: sequence_length
            # rewards: sequence_length
            for _ in range(self.config.sequence_length - len(self.recent_actions)):
                self.recent_states.append(self.dummy_state)
                self.recent_actions.append(random.randint(0, self.config.action_num - 1))
                self.recent_rewards.append(0)
            self.remote_memory.dynamics_add(
                {
                    "states": self.recent_states,
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

        _view_action = 4
        _view_sample = 3
        IMG_W = 64
        IMG_H = 64
        STR_H = 15
        PADDING = 4
        WIDTH = (IMG_W + PADDING) * _view_action + 5
        HEIGHT = (IMG_H + PADDING + STR_H) * (_view_sample + 1) + STR_H * 2 + 5

        if self.viewer is None:
            self.viewer = Viewer(WIDTH, HEIGHT)
        self.viewer.draw_fill(color=(0, 0, 0))

        # --- vae
        s = tf.concat([self.z, self.prev_hidden_state], axis=1)
        pred_state = self.parameter.model.decoder(s)[0].numpy()
        rmse = np.sqrt(np.mean((self.state - pred_state) ** 2))

        img1 = self.state * 255
        img2 = pred_state * 255

        self.viewer.draw_text(0, 0, "original", color=(255, 255, 255))
        self.viewer.draw_image_rgb_array(0, STR_H, img1)
        self.viewer.draw_text(IMG_W + PADDING, 0, f"decode(RMSE: {rmse:.5f})", color=(255, 255, 255))
        self.viewer.draw_image_rgb_array(IMG_W + PADDING, STR_H, img2)

        # 横にアクション後の結果を表示
        for i, a in enumerate(self.get_valid_actions()):
            if i > _view_action:
                break

            h, _ = self.parameter.model.one_step_transition(self.z, a, self.prev_hidden_state)

            self.viewer.draw_text(
                (IMG_W + PADDING) * i, 20 + IMG_H, f"action {env.action_to_str(a)}", color=(255, 255, 255)
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
                self.viewer.draw_text(x, y, f"{reward:.3f}", color=(255, 255, 255))
                self.viewer.draw_image_rgb_array(x, y + STR_H, n_img)

        return self.viewer.get_rgb_array()
