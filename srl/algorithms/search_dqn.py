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
from srl.rl.functions.common import render_discrete_action, symlog, twohot_encode
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, ExperienceReplayBufferConfig
from srl.rl.memories.priority_experience_replay import PriorityExperienceReplay
from srl.rl.models.image_block import ImageBlockConfig
from srl.rl.models.tf.distributions.bernoulli_dist_block import BernoulliDistBlock
from srl.rl.models.tf.distributions.normal_dist_block import NormalDistBlock
from srl.rl.models.tf.input_block import InputImageBlock
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
    h_size: int = 512
    #: <:ref:`ImageBlock`> This layer is only used when the input is an image.
    image_block: ImageBlockConfig = field(init=False, default_factory=lambda: ImageBlockConfig())

    reward_type: str = "twohot"  # "linear" "twohot"
    #: reward_typeが"twohot"の時のみ有効、bins
    reward_twohot_bins: int = 255
    #: reward_typeが"twohot"の時のみ有効、low
    reward_twohot_low: int = -5
    #: reward_typeが"twohot"の時のみ有効、high
    reward_twohot_high: int = 5

    target_q_int_update_interval: int = 1000
    target_q_ext_update_interval: int = 1000

    episodic_memory_capacity: int = 300_000

    #: [episodic] k
    episodic_count_max: int = 10
    #: [episodic] epsilon
    episodic_epsilon: float = 0.001
    #: [episodic] cluster_distance
    episodic_cluster_distance: float = 0.008

    epsilon = 0.1
    test_epsilon = 0.0
    policy_prob_int: float = 0.9

    search_rate: float = 0.5

    lr_feat: float = 0.0001  # type: ignore , type OK
    lr_mdp: float = 0.0001  # type: ignore , type OK
    lr_lifelong: float = 0.00001  # type: ignore , type OK
    lr_q_ext: float = 8e-5  # type: ignore , type OK
    lr_q_int: float = 8e-5  # type: ignore , type OK

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
    enable_train_int: bool = True
    enable_train_ext: bool = True
    enable_train_sample_ext: bool = True

    num_mixture: int = 20

    enable_train_model: bool = True
    enable_train_actor: bool = True
    enable_train_critic: bool = True
    batch_length: int = 50
    target_critic_update_interval: int = 100
    reinforce_rate: float = 0.5  # type: ignore , type OK
    entropy_rate: float = 0.001  # type: ignore , type OK
    reinforce_baseline: str = "v"  # "v"

    # Behavior
    discount: float = 0.999
    disclam: float = 0.95
    horizon: int = 15
    critic_estimation_method: str = "dreamer_v2"  # "simple" or "dreamer" or "dreamer_v2"

    def __post_init__(self):
        super().__post_init__()

        # self.lr_model: SchedulerConfig = SchedulerConfig(cast(float, self.lr_model))
        # self.lr_critic: SchedulerConfig = SchedulerConfig(cast(float, self.lr_critic))
        # self.lr_actor: SchedulerConfig = SchedulerConfig(cast(float, self.lr_actor))
        # self.reinforce_rate: SchedulerConfig = SchedulerConfig(cast(float, self.reinforce_rate))
        # self.entropy_rate: SchedulerConfig = SchedulerConfig(cast(float, self.entropy_rate))

    def get_processors(self) -> List[ObservationProcessor]:
        return [
            ImageProcessor(
                image_type=EnvTypes.GRAY_2ch,
                resize=(84, 84),
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
        return "SearchDQN"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_memory()


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
        self.config = config

        # --- input
        self.in_img_block = None
        if config.observation_type == RLTypes.IMAGE:
            self.in_img_block = InputImageBlock(config.observation_shape, config.env_observation_type)
            self.img_block = config.image_block.create_block_tf()
            self.share_layers = [
                kl.Flatten(),
                kl.LayerNormalization(),
                kl.Dense(config.h_size, activation="relu"),
            ]
        else:
            self.share_layers = [
                kl.Flatten(),
                kl.Dense(config.h_size, activation="relu"),
                kl.LayerNormalization(),
                kl.Dense(config.h_size, activation="relu"),
            ]

        self.act_layers = [
            # kl.Dense(config.z_size, activation="relu"),
            # kl.LayerNormalization(),
            kl.Dense(config.h_size, activation="relu"),
            kl.Dense(config.action_num, kernel_initializer="zeros"),
        ]

        self.loss_func = keras.losses.CategoricalCrossentropy(from_logits=True)

        # --- build
        self._in_shape = config.observation_shape
        z1, z2, a = self(
            [
                np.zeros((1,) + self._in_shape),
                np.zeros((1,) + self._in_shape),
            ]
        )
        self.out_size: int = z1.shape[1]

    def call(self, x, training=False):
        z1 = self.call_feature(x[0], training=training)
        z2 = self.call_feature(x[1], training=training)

        a = tf.concat([z1, z2], axis=1)
        for h in self.act_layers:
            a = h(a, training=training)

        return z1, z2, a

    def call_feature(self, x, training=False):
        if self.in_img_block is not None:
            x = self.in_img_block(x, training=training)
            x = self.img_block(x, training=training)
            for h in self.share_layers:
                x = h(x, training=training)
        else:
            for h in self.share_layers:
                x = h(x, training=training)
        return x

    @tf.function
    def compute_train_loss(self, state, next_state, action):
        z1, z2, a = self([state, next_state], training=True)
        loss = self.loss_func(action, a)
        return loss, z1, z2

    def summary(self, name="", **kwargs):
        # if self.in_img_block is not None:
        #    x = self.in_img_block(x, training=training)
        #    x = self.img_block(x, training=training)

        x = [
            kl.Input(shape=self._in_shape1, name="state"),
            kl.Input(shape=self._in_shape1, name="next_state"),
            kl.Input(shape=self._in_shape2, name="action"),
        ]
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x), name=name)
        return model.summary(**kwargs)


class _TransModel(keras.Model):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.h_layers = [
            kl.LayerNormalization(),
            kl.Dense(32, activation="relu"),
        ]

        # --- 決定的遷移と確率的遷移
        self.select_num = 2
        self.selector_layers = [
            kl.LayerNormalization(),
            kl.Dense(32, activation="relu"),
            kl.Dense(self.select_num, kernel_initializer="zeros"),
        ]

        # disc
        self.next_disc_layers = [
            kl.LayerNormalization(),
            kl.Dense(16, activation="relu"),
        ]

        # cat
        self.cat_units = 4
        self.cat_classes = 4
        self.next_cat_layers = [
            kl.LayerNormalization(),
            kl.Dense(16, activation="relu"),
            kl.Dense(self.cat_units * self.cat_classes, kernel_initializer="zeros"),
        ]

        # --- next
        self.next_out_layers = [
            kl.LayerNormalization(),
            kl.Dense(32, activation="relu"),
            kl.Dense(config.z_size),
        ]

        # --- reward
        self.reward_out_layers = [
            kl.LayerNormalization(),
            kl.Dense(32, activation="relu"),
            kl.Dense(1),
        ]

        # --- build
        self(
            [
                tf.zeros((1, config.z_size)),
                tf.zeros((1, config.action_num)),
                None,
            ]
        )

        self.loss_func_r = keras.losses.MeanSquaredError()

    def call(self, x, training=False):
        z = x[0]
        action = x[1]
        batch = z.shape[0]

        x = tf.concat([x, action], axis=-1)
        for h in self.h_layers:
            x = h(x, training=training)

        # --- trans
        select = x
        for h in self.selector_layers:
            select = h(select)

        # straight select
        sample = tf.random.categorical(select, 1)
        sample = tf.one_hot(tf.squeeze(sample, 1), self.select_num)
        probs = tf.nn.softmax(select)
        select = sample + probs - tf.stop_gradient(probs)

        # next disc
        disc = x
        for h in self.next_disc_layers:
            disc = h(disc)

        # next cat
        cat = x
        for h in self.next_cat_layers:
            cat = h(cat)
        # (batch, units*cat) -> (batch, units, cat) -> (batch, units*cat)
        cat = tf.reshape(cat, (batch, self.cat_units, self.cat_classes))
        cat = tf.nn.softmax(cat, axis=-1)
        cat = tf.reshape(cat, (batch, self.cat_units * self.cat_classes))

        # --- next
        disc = tf.math.multiply(disc, tf.expand_dims(select[:, 0], axis=-1))
        cat = tf.math.multiply(cat, tf.expand_dims(select[:, 1], axis=-1))
        x = tf.concat([disc, cat], axis=-1)
        n = x
        for h in self.next_out_layers:
            n = h(n)
        r = x
        for h in self.reward_out_layers:
            r = h(r)

        return n, r

    def sample(self, batch_size: int):
        z = np.random.normal(size=(batch_size, self.config.z_size)).astype(np.float32)

        onehot_act = []
        for _ in range(batch_size):
            act = random.randint(0, self.config.action_num - 1)
            onehot_act.append(np.identity(self.config.action_num)[act])
        onehot_act = np.asarray(onehot_act, dtype=np.float32)

        n_z, r = self.call([z, onehot_act])
        return z, onehot_act, n_z, r

    # @tf.functions
    def compute_train_loss(self, z, action, next_z, reward, done):
        n_z, r = self([z, action], training=True)
        n_loss = self.loss_func_n(next_z, n_z)
        r_loss = self.loss_func_r(reward, r)
        return n_loss, r_loss

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
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.h_layers = [
            kl.Dense(config.h_size, activation="relu"),
            kl.Dense(config.h_size, activation="relu"),
            kl.Dense(1),
        ]

        if config.reward_type == "linear":
            self.reward_layers.append(kl.Dense(1))
            self.loss_reward_func = keras.losses.MeanSquaredError()
        elif config.reward_type == "twohot":
            self.reward_layers.append(kl.Dense(config.reward_twohot_bins))
            self.loss_reward_func = keras.losses.CategoricalCrossentropy(from_logits=True)
        else:
            raise UndefinedError(config.reward_type)

        # --- build
        self(
            [
                tf.zeros((1, config.z_size)),
                tf.zeros((1, config.action_num)),
                tf.zeros((1, config.z_size)),
            ]
        )
        self.loss_func = keras.losses.MeanSquaredError()

    def call(self, x, training=False):
        z = x[0]
        action = x[1]
        n_z = x[2]
        x = tf.concat([z, action, n_z], axis=-1)
        for h in self.h_layers:
            x = h(x, training=training)
        return x

    @tf.function
    def compute_train_loss(self, z, action, n_z, reward):
        r = self([z, action, n_z], training=True)
        loss = self.loss_func(reward, r)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss


class _DoneModel(keras.Model):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.h_layers = [
            kl.Dense(config.h_size, activation="relu"),
            kl.Dense(config.h_size, activation="relu"),
            kl.Dense(1),
        ]

        # --- build
        self(
            [
                tf.zeros((1, config.z_size)),
                tf.zeros((1, config.action_num)),
                tf.zeros((1, config.z_size)),
            ]
        )
        self.loss_func = keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, x, training=False):
        z = x[0]
        action = x[1]
        n_z = x[2]
        x = tf.concat([z, action, n_z], axis=-1)
        for h in self.h_layers:
            x = h(x, training=training)
        return x

    @tf.function
    def compute_train_loss(self, z, action, n_z, done):
        d = self([z, action, n_z], training=True)
        loss = self.loss_func(done, d)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss


class _QNetwork(keras.Model):
    def __init__(self, config: Config, in_size: int, is_ext: bool):
        super().__init__()

        self.h_layers = [
            kl.Dense(config.h_size, activation="relu"),
            kl.Dense(config.h_size, activation="relu"),
        ]

        if is_ext:
            self.h_layers.append(
                kl.Dense(
                    config.action_num,
                    kernel_initializer="truncated_normal",
                    bias_initializer="truncated_normal",
                )
            )
        else:
            self.h_layers.append(
                kl.Dense(
                    config.action_num,
                    kernel_initializer="zeros",
                    bias_initializer="zeros",
                )
            )

        self.loss_func = keras.losses.Huber()

        # build
        self._in_shape = (in_size,)
        self.build((None,) + self._in_shape)

    def call(self, x, training=False):
        # UVFAは使えない
        for h in self.h_layers:
            x = h(x, training=training)
        return x

    @tf.function
    def compute_train_loss(self, z, onehot_action, target_q):
        q = self(z)
        q = tf.reduce_sum(q * onehot_action, axis=1)
        loss = self.loss_func(target_q, q)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss

    def summary(self, name="", **kwargs):
        x = kl.Input(shape=self._in_shape)
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x), name=name)
        return model.summary(**kwargs)


class _LifelongNetwork(keras.Model):
    def __init__(self, config: Config, in_size: int):
        super().__init__()

        # hidden
        self.h_layers = [
            kl.Dense(config.h_size, activation="relu"),
            kl.Dense(config.h_size, activation="relu"),
            kl.LayerNormalization(),  # last
        ]

        # build
        self._in_shape = (in_size,)
        self.build((None,) + self._in_shape)
        self.loss_func = keras.losses.MeanSquaredError()

    def call(self, x, training=False):
        for h in self.h_layers:
            x = h(x, training=training)
        return x

    @tf.function
    def compute_train_loss(self, z, target_val):
        val = self(z, training=True)
        loss = self.loss_func(target_val, val)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss

    def summary(self, name: str = "", **kwargs):
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

        self.feature = _SiameseNetwork(self.config)

        self.trans_model = _TransModel(self.config, self.feature.out_size)
        self.reward_model = _RewardModel(self.config, self.feature.out_size)
        self.done_model = _DoneModel(self.config, self.feature.out_size)

        self.q_ext_online = _QNetwork(self.config, self.feature.out_size)
        self.q_ext_target = _QNetwork(self.config, self.feature.out_size)
        self.q_ext_target.set_weights(self.q_ext_online.get_weights())
        self.q_int_online = _QNetwork(self.config, self.feature.out_size)
        self.q_int_target = _QNetwork(self.config, self.feature.out_size)
        self.q_int_target.set_weights(self.q_int_online.get_weights())

        self.lifelong_train = _LifelongNetwork(self.config, self.feature.out_size)
        self.lifelong_target = _LifelongNetwork(self.config, self.feature.out_size)

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
        self.feature.summary("feature", **kwargs)
        self.q_ext_online.summary("q_ext", **kwargs)
        self.q_int_online.summary("q_int", **kwargs)
        self.lifelong_train.summary("lifelong", **kwargs)


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        # if compare_less_version(tf.__version__, "2.11.0"):
        #    self._model_opt = keras.optimizers.Adam(learning_rate=self.lr_sch_model.get_rate())
        #    self._critic_opt = keras.optimizers.Adam(learning_rate=self.lr_sch_critic.get_rate())
        #    self._actor_opt = keras.optimizers.Adam(learning_rate=self.lr_sch_actor.get_rate())
        # else:
        #    pass
        self.opt_feat = keras.optimizers.legacy.Adam(learning_rate=self.config.lr_feat)
        self.opt_mdp = keras.optimizers.legacy.Adam(learning_rate=self.config.lr_mdp)
        self.opt_lifelong = keras.optimizers.legacy.Adam(learning_rate=self.config.lr_lifelong)
        self.opt_q_ext = keras.optimizers.legacy.Adam(learning_rate=self.config.lr_q_ext)
        self.opt_q_int = keras.optimizers.legacy.Adam(learning_rate=self.config.lr_q_int)

        self.sync_count = 0

        self.prev_disc_loss = np.inf
        self.prev_gen_loss = np.inf

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return

        state, action, n_state, reward_ext, reward_int, terminated = zip(
            *self.memory.sample(self.batch_size, self.train_count)
        )
        state = np.asarray(state)
        action = np.asarray(action)
        n_state = np.asarray(n_state)
        reward_ext = np.asarray(reward_ext)[..., np.newaxis]
        reward_int = np.asarray(reward_int)[..., np.newaxis]
        terminated = np.asarray(terminated)[..., np.newaxis]

        # --- feature
        if self.config.enable_train_feature:
            with tf.GradientTape() as tape:
                loss, z, n_z = self.parameter.feature.compute_train_loss(
                    state,
                    n_state,
                    action,
                )
            grad = tape.gradient(loss, self.parameter.feature.trainable_variables)
            self.opt_feat.apply_gradients(zip(grad, self.parameter.feature.trainable_variables))
            self.train_info["feat_loss"] = loss.numpy()
        else:
            z = self.parameter.feature.call_feature(state)
            n_z = self.parameter.feature.call_feature(n_state)

        if self.config.enable_train_mdp:
            if self.config.reward_type == "twohot":
                reward = symlog(reward_ext)
                reward = twohot_encode(
                    np.squeeze(reward, axis=1),
                    self.config.reward_twohot_bins,
                    self.config.reward_twohot_low,
                    self.config.reward_twohot_high,
                )
            else:
                reward = symlog(reward_ext)
                reward = reward_ext

            with tf.GradientTape() as tape:
                loss_t, pred_n_z = self.parameter.trans_model.compute_train_loss(z, action, n_z)
                loss_r = self.parameter.reward_model.compute_train_loss(
                    z, action, tf.stop_gradient(pred_n_z), reward_ext
                )
                loss_d = self.parameter.done_model.compute_train_loss(
                    z, action, tf.stop_gradient(pred_n_z), terminated
                )
                loss = loss_t + loss_r + loss_d
            vars = [
                self.parameter.trans_model.trainable_variables,
                self.parameter.reward_model.trainable_variables,
                self.parameter.done_model.trainable_variables,
            ]
            grads = tape.gradient(loss, vars)
            for i in range(len(vars)):
                self.opt_mdp.apply_gradients(zip(grads[i], vars[i]))
            self.train_info["mdp_t_loss"] = loss_t.numpy()
            self.train_info["mdp_r_loss"] = loss_r.numpy()
            self.train_info["mdp_d_loss"] = loss_d.numpy()

        if self.config.enable_train_int:
            # --- lifelong
            target_val = self.parameter.lifelong_target(z)
            with tf.GradientTape() as tape:
                train_val = self.parameter.lifelong_train(z, training=True)
                loss = tf.reduce_mean(tf.square(train_val - target_val))
            grad = tape.gradient(loss, self.parameter.lifelong_train.trainable_variables)
            self.opt_lifelong.apply_gradients(zip(grad, self.parameter.lifelong_train.trainable_variables))
            self.train_info["lifelong_loss"] = loss.numpy()

            # --- Q int
            n_q_target = self.parameter.q_int_target(n_z).numpy()
            n_q = self.parameter.q_int_online(n_z).numpy()
            probs = np.full_like(n_q, (1 - self.config.policy_prob_int) / self.config.action_num)
            max_idx = np.identity(self.config.action_num)[np.argmax(n_q, axis=-1)]
            probs += max_idx * self.config.policy_prob_int
            n_q_target = np.sum(n_q_target * probs, axis=-1)[..., np.newaxis]
            target_q = reward_int + (1 - terminated) * self.config.discount * n_q_target

            with tf.GradientTape() as tape:
                loss = self.parameter.q_int_online.compute_train_loss(z, action, target_q)
            grad = tape.gradient(loss, self.parameter.q_int_online.trainable_variables)
            self.opt_q_int.apply_gradients(zip(grad, self.parameter.q_int_online.trainable_variables))
            self.train_info["q_int_loss"] = loss.numpy()

            if self.train_count % self.config.target_q_int_update_interval == 1:
                self.parameter.q_int_target.set_weights(self.parameter.q_int_online.get_weights())

        # --- Q ext
        if self.config.enable_train_ext:
            n_q_target = self.parameter.q_ext_target(n_z).numpy()
            n_q = self.parameter.q_ext_online(n_z).numpy()
            n_act_idx = np.argmax(n_q, axis=-1)
            maxq = n_q_target[np.arange(self.config.batch_size), n_act_idx][..., np.newaxis]
            target_q = reward_ext + (1 - terminated) * self.config.discount * maxq

            with tf.GradientTape() as tape:
                loss = self.parameter.q_ext_online.compute_train_loss(z, action, target_q)
            grad = tape.gradient(loss, self.parameter.q_ext_online.trainable_variables)
            self.opt_q_ext.apply_gradients(zip(grad, self.parameter.q_ext_online.trainable_variables))
            self.train_info["q_ext_loss"] = loss.numpy()

            if self.train_count % self.config.target_q_ext_update_interval == 1:
                self.parameter.q_ext_target.set_weights(self.parameter.q_ext_online.get_weights())

        # --- Q ext horizon

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

    def on_start(self, worker: WorkerRun) -> None:
        if self.training:
            self.epsilon = self.config.epsilon
        else:
            self.epsilon = self.config.test_epsilon

    def on_reset(self, worker: WorkerRun) -> dict:
        self.episodic_memory = collections.deque(maxlen=self.config.episodic_memory_capacity)
        return {}

    def policy(self, worker: WorkerRun) -> Tuple[Any, dict]:
        if random.random() < self.epsilon:
            action = np.random.randint(0, self.config.action_num, size=1)
        else:
            z = self.parameter.feature.call_feature(worker.state[np.newaxis, ...])
            q_ext = self.parameter.q_ext_online(z).numpy()
            q_int = self.parameter.q_int_online(z).numpy()
            q = (1 - self.config.search_rate) * q_ext + self.config.search_rate * q_int
            action = np.argmax(q, axis=-1)

        return int(action), {}

    def on_step(self, worker: WorkerRun) -> dict:
        if False:
            n_s = worker.state[np.newaxis, ...]
            n_z = self.parameter.feature.call_feature(n_s)
            episodic_reward = self._calc_episodic_reward(n_z, update=True)
            lifelong_reward = self._calc_lifelong_reward(n_z)
            reward_int = (episodic_reward * lifelong_reward)[0]
        reward_int = 0

        if not self.training:
            return {}

        # symlog TODO
        reward_ext = worker.reward

        self.memory.add(
            [
                worker.prev_state,
                np.identity(self.config.action_num, dtype=np.float32)[worker.prev_action],
                worker.state,
                reward_ext,
                reward_int,
                worker.done_type == DoneTypes.TERMINATED,
            ]
        )
        if worker.done_type == DoneTypes.TERMINATED:
            # MDP loop用
            for act in range(self.config.action_num):
                self.memory.add(
                    [
                        worker.state,
                        [1 if act == a else 0 for a in range(self.config.action_num)],
                        worker.state,
                        0,
                        -10,  # 使われないはず
                        True,
                    ]
                )
        return {}

    def _calc_episodic_reward(self, z, update: bool):
        k = self.config.episodic_count_max
        epsilon = self.config.episodic_epsilon
        cluster_distance = self.config.episodic_cluster_distance

        if len(self.episodic_memory) == 0:
            self.episodic_memory.append(z)
            return 0

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
        # reward = 1 / (np.sqrt(N) + 1)
        reward = -(np.sqrt(N + 1) + 1)
        return reward

    def _calc_lifelong_reward(self, z):
        rnd_target_val = self.parameter.lifelong_target(z)
        rnd_train_val = self.parameter.lifelong_train(z)

        # RMSE
        error = np.mean(np.square(rnd_target_val - rnd_train_val), axis=-1)
        error = np.sqrt(error)
        return np.minimum(error, 1) - 1

    def render_terminal(self, worker, **kwargs) -> None:
        z, n_z, a = self.parameter.feature(
            [
                worker.prev_state[np.newaxis, ...],
                worker.state[np.newaxis, ...],
                np.identity(self.config.action_num, dtype=np.float32)[worker.prev_action][np.newaxis, ...],
            ]
        )

        episodic_reward = self._calc_episodic_reward(z, update=False)
        lifelong_reward = self._calc_lifelong_reward(z)
        reward_int = episodic_reward * lifelong_reward
        print(f"reward_int: {reward_int}, episodic {episodic_reward}, lifelong {lifelong_reward}, ")
        print(f"reward_ext: {r}")
        print(f"done      : {d.numpy()[0][0]*100:.1f}%")

        q_ext = self.parameter.q_ext_online(z).numpy()[0]
        q_int = self.parameter.q_int_online(z).numpy()[0]
        q = q_ext + q_int
        maxa = np.argmax(q, axis=-1)

        def _render_sub(a: int) -> str:
            s = f"{q[a]:7.3f}"
            s += f"{a:2d}: ext {q_ext[a]:7.3f}, int {q_int[a]:7.3f}"
            return s

        render_discrete_action(maxa, worker.env, self.config, _render_sub)
