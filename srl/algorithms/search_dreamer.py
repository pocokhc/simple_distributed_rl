import collections
import logging
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from srl.base.define import DoneTypes, EnvTypes, RLBaseTypes, RLTypes
from srl.base.env.env_run import EnvRun
from srl.base.exception import UndefinedError
from srl.base.rl.algorithms.discrete_action import DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer, RLWorker
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import ObservationProcessor
from srl.base.rl.registration import register
from srl.base.rl.worker_run import WorkerRun
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, ExperienceReplayBufferConfig
from srl.rl.memories.priority_experience_replay import PriorityExperienceReplay
from srl.rl.models.image_block import ImageBlockConfig

from srl.rl.processors.image_processor import ImageProcessor
from srl.rl.schedulers.scheduler import SchedulerConfig
from srl.utils.common import compare_less_version

kl = keras.layers
tfd = tfp.distributions

logger = logging.getLogger(__name__)


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig, ExperienceReplayBufferConfig):
    z_size: int = 512
    #: <:ref:`ImageBlock`> This layer is only used when the input is an image.
    image_block: ImageBlockConfig = field(init=False, default_factory=lambda: ImageBlockConfig())

    # Model
    deter_size: int = 600
    stoch_size: int = 32
    reward_layer_sizes: Tuple[int, ...] = (400, 400, 400, 400)
    discount_layer_sizes: Tuple[int, ...] = (400, 400, 400, 400)
    critic_layer_sizes: Tuple[int, ...] = (400, 400, 400, 400)
    actor_layer_sizes: Tuple[int, ...] = (400, 400, 400, 400)
    dense_act: Any = "elu"
    cnn_act: Any = "relu"
    cnn_depth: int = 32
    free_nats: float = 3.0
    kl_scale: float = 0.1
    fixed_variance: bool = False
    vae_discrete: bool = True
    kl_balancing_rate: float = 0.8
    h_target: float = 0.95

    # Training
    enable_train_feature: bool = True
    enable_train_mdp: bool = True
    enable_train_gan: bool = True
    enable_train_int: bool = True
    enable_train_ext: bool = True
    enable_train_sample_ext: bool = True

    num_mixture: int = 20

    enable_train_model: bool = True
    enable_train_actor: bool = True
    enable_train_critic: bool = True
    batch_length: int = 50
    lr_model: float = 6e-4  # type: ignore , type OK
    lr_critic: float = 8e-5  # type: ignore , type OK
    lr_actor: float = 8e-5  # type: ignore , type OK
    target_critic_update_interval: int = 100
    reinforce_rate: float = 0.5  # type: ignore , type OK
    entropy_rate: float = 0.001  # type: ignore , type OK
    reinforce_baseline: str = "v"  # "v"

    # Behavior
    discount: float = 0.999
    disclam: float = 0.95
    horizon: int = 15
    critic_estimation_method: str = "dreamer_v2"  # "simple" or "dreamer" or "dreamer_v2"

    # debug action
    epsilon: float = 0.0

    def __post_init__(self):
        super().__post_init__()

        self.lr_model: SchedulerConfig = SchedulerConfig(cast(float, self.lr_model))
        self.lr_critic: SchedulerConfig = SchedulerConfig(cast(float, self.lr_critic))
        self.lr_actor: SchedulerConfig = SchedulerConfig(cast(float, self.lr_actor))
        self.reinforce_rate: SchedulerConfig = SchedulerConfig(cast(float, self.reinforce_rate))
        self.entropy_rate: SchedulerConfig = SchedulerConfig(cast(float, self.entropy_rate))

    def get_processors(self) -> List[ObservationProcessor]:
        return [
            ImageProcessor(
                image_type=EnvTypes.COLOR,
                resize=(64, 64),
                enable_norm=True,
            )
        ]

    def get_base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.CONTINUOUS

    def get_use_framework(self) -> str:
        return "tensorflow"

    def get_name(self) -> str:
        return "SearchDreamer"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_memory()

    def set_config_by_env(self, env: EnvRun) -> None:
        if self.experience_acquisition_method == "episode" and self.batch_length < env.max_episode_steps:
            s = "Learning may not be possible because 'batch_length' is shorter than the number of episode steps in env."
            s += f" batch_length={self.batch_length}, env={env.max_episode_steps}"
            logger.warning(s)


register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


class Memory(ExperienceReplayBuffer):
    pass


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _SiameseNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()
        self.z_size = config.z_size

        # --- input
        self.in_img_block = None
        if config.observation_type == RLTypes.IMAGE:
            self.in_img_block = InputImageBlock(config.observation_shape, config.env_observation_type)
            self.img_block = config.image_block.create_block_tf()
        self.flat_layer = kl.Flatten()

        self.share_layers = [
            kl.Dense(config.z_size * 2, activation="relu"),
            kl.LayerNormalization(),
            kl.Dense(config.z_size * 2, activation="relu"),
            kl.Dense(config.z_size, activation="relu"),
        ]

        self.out_layers = [
            kl.Dense(config.z_size * 2, activation="relu"),
            kl.Dense(config.action_num),
        ]

        self.loss_func = keras.losses.CategoricalCrossentropy(from_logits=True)

        # --- build
        a, z1, z2 = self(
            [
                np.zeros((1,) + config.observation_shape),
                np.zeros((1,) + config.observation_shape),
            ]
        )
        self.out_size = z1.shape[1]

    def call(self, x, training=False):
        z1 = self.call_feature(x[0], training=training)
        z2 = self.call_feature(x[1], training=training)
        x = tf.concat([z1, z2], axis=1)
        for h in self.out_layers:
            x = h(x)
        return x, z1, z2

    def call_feature(self, x, training=False):
        if self.in_img_block is not None:
            x = self.in_img_block(x, training=training)
            x = self.img_block(x, training=training)
        x = self.flat_layer(x)

        for h in self.share_layers:
            x = h(x, training=training)
        return x

    # @tf.functions
    def compute_train_loss(self, state1, state2, onehot_action):
        act, z1, z2 = self([state1, state2])
        loss = self.loss_func(onehot_action, act)
        return loss, z1, z2

    def summary(self, name="", **kwargs):
        x = kl.Input(shape=self._input_shape)
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x), name=name)
        return model.summary(**kwargs)


class _Generator(keras.Model):
    def __init__(self, config: Config, in_size: int):
        super().__init__()

        self.noise_size = in_size

        self.h_layers = [
            kl.Dense(config.z_size * 4, activation="relu"),
            kl.LayerNormalization(),
            kl.Dense(config.z_size * 2, activation="relu"),
            kl.Dense(config.z_size * 2, activation="relu"),
            kl.Dense(in_size),
        ]
        self.loss_func = keras.losses.Huber()

    def call(self, x):
        for h in self.h_layers:
            x = h(x)
        return x

    def sample(self, size: int):
        x = tf.random.uniform(shape=(size, self.noise_size), minval=-1, maxval=1)
        return self(x)


class _Discriminator(keras.Model):
    def __init__(self, config: Config, in_size: int):
        super().__init__()

        self.h_layers = [
            kl.Dense(config.z_size * 2, activation="relu"),
            kl.LayerNormalization(),
            kl.Dense(config.z_size * 2, activation="relu"),
            kl.Dense(1),
        ]
        self.loss_func = keras.losses.Huber()

    def call(self, x):
        for h in self.h_layers:
            x = h(x)
        return x


class _TransModel(keras.Model):
    def __init__(self, config: Config, in_size: int):
        super().__init__()
        self.num_mixture = config.num_mixture

        # MDN
        self.h_layers = [
            kl.Dense(config.z_size * 2, activation="relu"),
            kl.LayerNormalization(),
            kl.Dense(config.z_size * 2, activation="relu"),
            kl.Dense(self.num_mixture * 3),
        ]

        # --- build
        self(
            [
                tf.zeros((1, in_size)),
                tf.zeros((1, config.action_num)),
            ]
        )

    def call(self, x, training=False):
        z = x[0]
        action = x[1]

        x = tf.concat([z, action], axis=-1)
        for h in self.h_layers:
            x = h(x, training=training)

        pi = x[:, 0 : self.num_mixture]
        mu = x[:, self.num_mixture : self.num_mixture * 2]
        log_sigma = x[:, self.num_mixture * 2 : self.num_mixture * 3]

        return pi, mu, log_sigma

    # @tf.functions
    def compute_train_loss(self, z, action, n_z):
        pi, mu, log_sigma = self([z, action])

        # ソフトマックス関数のオーバーフロー防止
        # (https://leico.github.io/TechnicalNote/Math/deep_learning)
        pi = pi - tf.reduce_max(pi, axis=1, keepdims=True)

        # log softmax
        log_pi = pi - tf.math.log(tf.reduce_sum(tf.exp(pi), axis=1, keepdims=True))

        # los gauss
        log_gauss = -0.5 * (np.log(2 * np.pi) + 2 * log_sigma + (n_z - mu) ** 2 / tf.exp(log_sigma) ** 2)

        # loss
        loss = tf.reduce_sum(tf.exp(log_pi + log_gauss), axis=1, keepdims=True)
        loss = tf.maximum(loss, 1e-10)  # log(0) 回避
        return -tf.reduce_mean(tf.math.log(loss))

    def summary(self, config, **kwargs):
        in_stoch = kl.Input((config.stoch_size,))
        in_deter = kl.Input((config.deter_size,))
        in_action = kl.Input((config.action_num,))
        in_embed = kl.Input((32 * config.cnn_depth,))

        deter, prior = self.img_step(in_stoch, in_deter, in_action, _summary=True)
        post = self.obs_step(deter, in_embed, _summary=True)
        model = keras.Model(
            inputs=[in_stoch, in_deter, in_action, in_embed],
            outputs=post,
            name="RSSM",
        )
        return model.summary(**kwargs)


class _RewardModel(keras.Model):
    def __init__(self, config: Config, in_size: int):
        super().__init__()

        self.h_layers = [
            kl.Dense(config.z_size, activation="relu"),
            kl.LayerNormalization(),
            kl.Dense(config.z_size, activation="relu"),
            kl.Dense(1),
        ]

        # build
        self(
            [
                tf.zeros((1, in_size)),
                tf.zeros((1, config.action_num)),
                tf.zeros((1, in_size)),
            ]
        )
        self.loss_func = keras.losses.Huber()

    def call(self, x):
        z = x[0]
        act = x[1]
        n_z = x[2]
        x = tf.concat([z, act, n_z], axis=-1)
        for h in self.h_layers:
            x = h(x)
        return x

    def compute_train_loss(self, z, act, n_z, reward):
        r = self([z, act, n_z])
        return self.loss_func(reward, r)


class _DoneModel(keras.Model):
    def __init__(self, config: Config, in_size: int):
        super().__init__()

        self.h_layers = [
            kl.Dense(config.z_size, activation="relu"),
            kl.LayerNormalization(),
            kl.Dense(config.z_size, activation="relu"),
            kl.Dense(1),
        ]

        # build
        self(
            [
                tf.zeros((1, in_size)),
                tf.zeros((1, config.action_num)),
                tf.zeros((1, in_size)),
            ]
        )
        self.loss_func = keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, x):
        z = x[0]
        act = x[1]
        n_z = x[2]
        x = tf.concat([z, act, n_z], axis=-1)
        for h in self.h_layers:
            x = h(x)
        return x

    # @tf.function()
    def compute_train_loss(self, z, act, n_z, done):
        d = self([z, act, n_z])
        return self.loss_func(done, d)


class _QNetwork(keras.Model):
    def __init__(self, config: Config, in_size: int):
        super().__init__()

        self.h_layers = [
            kl.Dense(32, activation="relu"),
        ]

        self.out_layer = kl.Dense(
            config.action_num,
            kernel_initializer="truncated_normal",
            bias_initializer="truncated_normal",
        )

        # build
        self._in_shape = (config.z_size,)
        self.build((None,) + self._in_shape)

    # @tf.function()
    def call(self, x, training=False):
        # UVFAは使えない
        for h in self.h_layers:
            x = h(x, training=training)
        return self.out_layer(x, training=training)


class _LifelongNetwork(keras.Model):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        # input, 直接状態を使う
        self.in_img_block = None
        if config.observation_type == RLTypes.IMAGE:
            self.in_img_block = InputImageBlock(config.observation_shape, config.env_observation_type)
            self.img_block = config.image_block.create_block_tf()
        self.flat_layer = kl.Flatten()

        # hidden
        self.h_layers = [
            kl.Dense(32, activation="relu"),
            kl.LayerNormalization(),  # last
        ]

        # build
        self._in_shape = config.observation_shape
        self.build((None,) + self._in_shape)
        self.loss_func = keras.losses.MeanSquaredError()

    def call(self, x, training=False):
        if self.in_img_block is not None:
            x = self.in_img_block(x, training=training)
            x = self.img_block(x, training=training)
        x = self.flat_layer(x)
        for h in self.h_layers:
            x = h(x, training=training)
        return x

    @tf.function
    def compute_train_loss(self, state, target_val):
        val = self(state, training=True)
        loss = self.loss_func(target_val, val)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss

    def summary(self, name: str = "", **kwargs):
        if self.in_img_block is not None:
            self.in_img_block.init_model_graph()
            self.img_block.init_model_graph()
        self.hidden_block.init_model_graph()

        x = kl.Input(shape=self._in_shape)
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x), name=name)
        model.summary(**kwargs)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.feature_embedding = _SiameseNetwork(self.config)

        self.trans_model = _TransModel(self.config, self.feature_embedding.out_size)
        self.reward_model = _RewardModel(self.config, self.feature_embedding.out_size)
        self.done_model = _DoneModel(self.config, self.feature_embedding.out_size)

        self.generator = _Generator(self.config, self.feature_embedding.out_size)
        self.discriminator = _Discriminator(self.config, self.feature_embedding.out_size)

        self.q_ext_online = _QNetwork(self.config)
        self.q_ext_target = _QNetwork(self.config)
        self.q_ext_target.set_weights(self.q_ext_online.get_weights())
        self.q_int_online = _QNetwork(self.config)
        self.q_int_target = _QNetwork(self.config)
        self.q_int_target.set_weights(self.q_int_online.get_weights())

        self.lifelong_train = _LifelongNetwork(self.config)
        self.lifelong_target = _LifelongNetwork(self.config)

    def policy(self, z, epsilon, invalid_actions):
        acts = np.random.randint(0, self.action_num, size=z.shape[0])
        r = np.random.random(z.shape[0])
        r = np.where(r < epsilon, 1, 0)

        q_ext = self.q_ext_online(z).numpy()
        q_int = self.q_int_online(z).numpy()
        q = q_ext + q_int
        acts2 = np.argmax(q, axis=-1)

        a = acts * r + acts2 * (1 - r)
        return a

    def call_restore(self, data: Any, **kwargs) -> None:
        self.encode.set_weights(data[0])
        self.dynamics.set_weights(data[1])
        self.decode.set_weights(data[2])
        self.reward.set_weights(data[3])
        self.discount.set_weights(data[4])
        self.critic.set_weights(data[5])
        self.critic_target.set_weights(data[5])
        self.actor.set_weights(data[6])

    def call_backup(self, **kwargs) -> Any:
        return [
            self.encode.get_weights(),
            self.dynamics.get_weights(),
            self.decode.get_weights(),
            self.reward.get_weights(),
            self.discount.get_weights(),
            self.critic.get_weights(),
            self.actor.get_weights(),
        ]

    def summary(self, **kwargs):
        self.encode.summary("Encoder", **kwargs)
        self.dynamics.summary(self.config, **kwargs)
        self.decode.summary("Decoder", **kwargs)
        self.reward.summary("Reward", **kwargs)
        self.discount.summary("Discount", **kwargs)
        self.critic.summary("Critic", **kwargs)
        self.actor.summary("Actor", **kwargs)


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        if compare_less_version(tf.__version__, "2.11.0"):
            self._model_opt = keras.optimizers.Adam(learning_rate=self.lr_sch_model.get_rate())
            self._critic_opt = keras.optimizers.Adam(learning_rate=self.lr_sch_critic.get_rate())
            self._actor_opt = keras.optimizers.Adam(learning_rate=self.lr_sch_actor.get_rate())
        else:
            self._model_opt = keras.optimizers.legacy.Adam(learning_rate=self.lr_sch_model.get_rate())
            self._critic_opt = keras.optimizers.legacy.Adam(learning_rate=self.lr_sch_critic.get_rate())
            self._actor_opt = keras.optimizers.legacy.Adam(learning_rate=self.lr_sch_actor.get_rate())

            self.optimaizer_feat = keras.optimizers.legacy.Adam(learning_rate=self.lr_sch_model.get_rate())
            self.optimaizer_mdp = keras.optimizers.legacy.Adam(learning_rate=self.lr_sch_model.get_rate())
            self.optimaizer_gen = keras.optimizers.legacy.Adam(learning_rate=self.lr_sch_model.get_rate())
            self.optimaizer_disc = keras.optimizers.legacy.Adam(learning_rate=self.lr_sch_model.get_rate())
            self.optimizer_lifelong = keras.optimizers.legacy.Adam(learning_rate=self.lr_sch_model.get_rate())
            self.optimizer_q_int = keras.optimizers.legacy.Adam(learning_rate=self.lr_sch_model.get_rate())
            self.optimizer_q_ext = keras.optimizers.legacy.Adam(learning_rate=self.lr_sch_model.get_rate())

        self.sync_count = 0

        self.prev_disc_loss = np.inf
        self.prev_gen_loss = np.inf

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return

        state, onehot_action, n_state, reward_ext, reward_int, terminated = zip(
            *self.memory.sample(self.batch_size, self.train_count)
        )
        state = np.asarray(state)
        onehot_action = np.asarray(onehot_action)
        n_state = np.asarray(n_state)
        reward_ext = np.asarray(reward_ext)
        reward_int = np.asarray(reward_int)
        terminated = np.asarray(terminated)

        # --- feature
        if self.config.enable_train_feature:
            with tf.GradientTape() as tape:
                loss, z, n_z = self.parameter.feature_embedding.compute_train_loss(state, onehot_action, n_state)
            grad = tape.gradient(loss, self.parameter.feature_embedding.trainable_variables)
            self.optimaizer_feat.apply_gradients(zip(grad, self.parameter.feature_embedding.trainable_variables))
            self.train_info["feat_loss"] = loss
        else:
            z = self.parameter.feature_embedding.call_feature(state)
            n_z = self.parameter.feature_embedding.call_feature(n_state)

        # --- mdp
        if self.config.enable_train_mdp:
            with tf.GradientTape() as tape:
                loss_t = self.parameter.trans_model.compute_train_loss(z, onehot_action, n_z)
                loss_r = self.parameter.reward_model.compute_train_loss(z, onehot_action, n_z, reward_ext)
                loss_d = self.parameter.done_model.compute_train_loss(z, onehot_action, n_z, terminated)
                loss = loss_t + loss_r + loss_d
            vars = [
                self.parameter.trans_model.trainable_variables,
                self.parameter.reward_model.trainable_variables,
                self.parameter.done_model.trainable_variables,
            ]
            grads = tape.gradient(loss, vars)
            for i in range(len(vars)):
                self.optimaizer_mdp.apply_gradients(zip(grads[i], vars[i]))
            self.train_info["trans_loss"] = loss_t
            self.train_info["reward_loss"] = loss_r
            self.train_info["done_loss"] = loss_d

        # --- GAN
        if self.config.enable_train_gan:
            if self.prev_gen_loss <= self.prev_disc_loss:
                # Generator
                self.parameter.generator.trainable = True
                self.parameter.discriminator.trainable = False
                y_real = tf.ones((self.config.batch_size * 4, 1))
                with tf.GradientTape() as tape:
                    z_fake = self.parameter.generator.sample(self.config.batch_size * 4)
                    y_pred = self.parameter.discriminator(z_fake)
                    loss = self.parameter.generator.loss_func(y_real, y_pred)
                grad = tape.gradient(loss, self.parameter.generator.trainable_variables)
                self.optimaizer_gen.apply_gradients(zip(grad, self.parameter.generator.trainable_variables))
                self.train_info["gen_loss"] = loss
            else:
                # Discriminator
                self.parameter.generator.trainable = False
                self.parameter.discriminator.trainable = True
                y_real = tf.ones((self.config.batch_size * 2, 1))
                y_fake = tf.ones((self.config.batch_size * 2, 1))
                z_fake = self.parameter.generator.sample(self.config.batch_size * 2)
                z_disc = tf.concat([z, z_fake], axis=0)
                y_disc = tf.concat([y_real, y_fake], axis=0)
                with tf.GradientTape() as tape:
                    y_pred = self.parameter.discriminator(z_disc)
                    loss = self.parameter.discriminator.loss_func(y_disc, y_pred)
                grad = tape.gradient(loss, self.parameter.discriminator.trainable_variables)
                self.optimaizer_disc.apply_gradients(zip(grad, self.parameter.discriminator.trainable_variables))
                self.train_info["disc_loss"] = loss

        if self.config.enable_train_int:
            # --- lifelong
            target_val = self.parameter.lifelong_target(state)
            with tf.GradientTape() as tape:
                train_val = self.parameter.lifelong_train(state, training=True)
                loss = tf.reduce_mean(tf.square(train_val - target_val))
            grad = tape.gradient(loss, self.parameter.lifelong_train.trainable_variables)
            self.optimizer_lifelong.apply_gradients(zip(grad, self.parameter.lifelong_train.trainable_variables))
            self.train_info["lifelong_loss"] = loss.numpy()

            # --- Q int
            n_q_target = self.parameter.q_int_target(n_z).numpy()
            n_q = self.parameter.q_int_online(n_z).numpy()
            probs = np.full_like(n_q, (1 - self.config.epsilon) / self.config.action_num)
            max_idx = np.identity(self.config.action_num)[np.argmax(n_q, axis=-1)]
            probs += max_idx * self.config.epsilon
            n_q_target = np.sum(n_q_target * probs, axis=-1)[..., np.newaxis]
            target_q = reward_int + (1 - terminated) * self.config.discount * n_q_target

            with tf.GradientTape() as tape:
                loss = self.parameter.q_int_online.compute_train_loss(z, onehot_action, target_q)
            grad = tape.gradient(loss, self.parameter.q_int_online.trainable_variables)
            self.optimizer_q_int.apply_gradients(zip(grad, self.parameter.q_int_online.trainable_variables))
            self.train_info["q_int_loss"] = loss.numpy()

            if self.train_count % self.config.target_q_int_update_interval == 1:
                self.parameter.q_int_target.set_weights(self.parameter.q_int_online.get_weights())

        # --- Q ext
        if self.config.enable_train_ext:
            n_q_target = self.parameter.q_ext_target(n_z).numpy()
            n_q = self.parameter.q_ext_online(n_z).numpy()
            n_act_idx = np.argmax(n_q, axis=-1)
            maxq = n_q_target[np.arange(self.config.batch_size, n_act_idx)][..., np.newaxis]
            target_q = reward_ext + (1 - terminated) * self.config.discount * maxq

            with tf.GradientTape() as tape:
                loss = self.parameter.q_ext_online.compute_train_loss(z, onehot_action, target_q)
            grad = tape.gradient(loss, self.parameter.q_ext_online.trainable_variables)
            self.optimizer_q_ext.apply_gradients(zip(grad, self.parameter.q_ext_online.trainable_variables))
            self.train_info["q_ext_loss"] = loss.numpy()

            if self.train_count % self.config.target_q_ext_update_interval == 1:
                self.parameter.q_ext_target.set_weights(self.parameter.q_ext_online.get_weights())

        # --- Q exp sample
        if self.config.enable_train_sample_ext:
            z = self.parameter.generator.sample(self.config.batch_size)
            act TODO
            n_z = self.parameter.trans_model(z, act)
            reward = self.parameter.reward_model([z, act, n_z])
            done = self.parameter.done_model([z, act, n_z])


        self.train_count += 1


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter
        self.screen = None

    def on_reset(self, worker: WorkerRun) -> dict:
        self.episodic_memory = collections.deque(maxlen=self.config.episodic_memory_capacity)
        return {}

    def policy(self, worker: WorkerRun) -> Tuple[Any, dict]:
        self.state = worker.state

        if random.random() < self.config.epsilon:
            action = np.random.randint(0, self.config.action_num, size=1)
        else:
            z, _, _ = self.parameter.feature_embedding.call_feature(self.state[np.newaxis, ...])
            if self.config.search_mode:
                action = self.parameter.policy
            else:
                q_ext = self.parameter.q_ext_online(z).numpy()
                q_int = self.parameter.q_int_online(z).numpy()
                q = q_ext
                action = np.argmax(q, axis=-1)

        self.onehot_action = np.identity(self.config.action_num, dtype=np.float32)[action][0]

        return int(action), {}

    def on_step(self, worker: WorkerRun) -> dict:
        n_s = worker.state[np.newaxis, ...]
        n_z, _, _ = self.parameter.feature_embedding.call_feature(n_s)
        episodic_reward = self._calc_episodic_reward(n_z, update=True)
        lifelong_reward = self._calc_lifelong_reward(n_z)
        reward_int = episodic_reward * lifelong_reward

        if not self.training:
            return {}

        batch = [
            self.state,
            self.onehot_action,
            worker.state,
            worker.reward,
            reward_int,
            worker.done_type == DoneTypes.TERMINATED,
        ]
        self.memory.add(batch)
        return {}

    def _calc_episodic_reward(self, z, update: bool):
        k = self.config.episodic_count_max
        epsilon = self.config.episodic_epsilon
        cluster_distance = self.config.episodic_cluster_distance

        if len(self.episodic_memory) == 0:
            self.episodic_memory.append(z)
            return 1

        # エピソードメモリ内の全要素とユークリッド距離を求める
        euclidean_list = [np.linalg.norm(m - z, ord=2) for m in self.episodic_memory]

        # エピソードメモリに制御可能状態を追加
        if update:
            self.episodic_memory.append(z)

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
        reward = 1 / (np.sqrt(N) + 1)
        return reward

    def _calc_lifelong_reward(self, z):
        rnd_target_val = self.parameter.lifelong_target(z)
        rnd_train_val = self.parameter.lifelong_train(z)

        # RMSE
        error = np.mean(np.square(rnd_target_val - rnd_train_val), axis=-1)
        error = np.sqrt(error)
        return error

    def render_terminal(self, worker, **kwargs) -> None:
        pass

    def render_rgb_array(self, worker, **kwargs) -> Optional[np.ndarray]:
        if self.config.env_observation_type != EnvTypes.COLOR:
            return None

        from srl.utils import pygame_wrapper as pw

        _view_action = 4
        _view_sample = 3
        IMG_W = 64
        IMG_H = 64
        STR_H = 15
        PADDING = 4
        WIDTH = (IMG_W + PADDING) * _view_action + 5
        HEIGHT = (IMG_H + PADDING + STR_H * 3) * (_view_sample + 1) + 5

        if self.screen is None:
            self.screen = pw.create_surface(WIDTH, HEIGHT)
        pw.draw_fill(self.screen, color=(0, 0, 0))

        if self.feat is None:
            self._rssm_step()

        # --- decode
        pred_state = self.parameter.decode(self.feat).mode()[0].numpy()  # type:ignore , ignore check "None"
        rmse = np.sqrt(np.mean((self.state - pred_state) ** 2))

        pred_reward = self.parameter.reward(self.feat).mode()[0][0].numpy()  # type:ignore , ignore check "None"
        pred_value = self.parameter.critic(self.feat).mode()[0][0].numpy()
        pred_discount = self.parameter.discount(self.feat).mode()[0][0].numpy()
        _, policy_logits = self.parameter.actor(self.feat, return_logits=True)  # type:ignore , ignore check "None"
        policy_logits = policy_logits[0].numpy()

        img1 = self.state * 255
        img2 = pred_state * 255

        pw.draw_text(self.screen, 0, 0, "original", color=(255, 255, 255))
        pw.draw_image_rgb_array(self.screen, 0, STR_H, img1)
        pw.draw_text(self.screen, IMG_W + PADDING, 0, f"decode(RMSE: {rmse:.4f})", color=(255, 255, 255))
        pw.draw_image_rgb_array(self.screen, IMG_W + PADDING, STR_H, img2)

        pw.draw_text(self.screen, IMG_W * 2 + PADDING + 10, 10, f"reward: {pred_reward:.4f}", color=(255, 255, 255))
        pw.draw_text(self.screen, IMG_W * 2 + PADDING + 10, 20, f"V     : {pred_value:.4f}", color=(255, 255, 255))
        pw.draw_text(self.screen, IMG_W * 2 + PADDING + 10, 30, f"dis   : {pred_discount:.4f}", color=(255, 255, 255))

        # 横にアクション後の結果を表示
        for a in range(self.config.action_num):
            if a in self.get_invalid_actions():
                continue
            if a > _view_action:
                break
            pw.draw_text(
                self.screen,
                (IMG_W + PADDING) * a,
                20 + IMG_H,
                f"{worker.env.action_to_str(a)}({policy_logits[a]:.2f})",
                color=(255, 255, 255),
            )

            action = tf.one_hot([a], self.config.action_num, axis=1)
            deter, prior = self.parameter.dynamics.img_step(self.stoch, self.deter, action)
            feat = tf.concat([prior["stoch"], deter], axis=1)

            next_state_dist = self.parameter.decode(feat)
            reward_dist = self.parameter.reward(feat)
            critic_dist = self.parameter.critic(feat)
            discount_dist = self.parameter.discount(feat)

            # 縦にいくつかサンプルを表示
            for j in range(_view_sample):
                if j == 0:
                    next_state = next_state_dist.mode()
                    reward = reward_dist.mode()
                    value = critic_dist.mode()
                    discount = discount_dist.mode()
                else:
                    next_state = next_state_dist.sample()
                    reward = reward_dist.sample()
                    value = critic_dist.sample()
                    discount = discount_dist.sample()

                n_img = next_state[0].numpy() * 255
                reward = reward.numpy()[0][0]
                value = value.numpy()[0][0]
                discount = discount.numpy()[0][0]

                x = (IMG_W + PADDING) * a
                y = 20 + IMG_H + STR_H + (IMG_H + PADDING + STR_H * 3) * j
                pw.draw_text(self.screen, x, y + STR_H * 0, f"r={reward:.3f}", color=(255, 255, 255))
                pw.draw_text(self.screen, x, y + STR_H * 1, f"V={value:.3f}", color=(255, 255, 255))
                pw.draw_text(self.screen, x, y + STR_H * 2, f"d={discount:.3f}", color=(255, 255, 255))
                pw.draw_image_rgb_array(self.screen, x, y + STR_H * 3, n_img)

        return pw.get_rgb_array(self.screen)
