import logging
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from srl.base.define import EnvObservationTypes, RLTypes
from srl.base.env.env_run import EnvRun
from srl.base.rl.algorithms.discrete_action import DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.rl.registration import register
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, ExperienceReplayBufferConfig
from srl.rl.processors.image_processor import ImageProcessor
from srl.rl.schedulers.scheduler import SchedulerConfig
from srl.utils.common import compare_less_version

kl = keras.layers
tfd = tfp.distributions

logger = logging.getLogger(__name__)

"""
paper: https://arxiv.org/abs/2010.02193
ref: https://github.com/danijar/dreamerv2/tree/07d906e9c4322c6fc2cd6ed23e247ccd6b7c8c41
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig, ExperienceReplayBufferConfig):
    capacity: int = 100_000
    memory_warmup_size: int = 1000

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
    enable_train_model: bool = True
    enable_train_actor: bool = True
    enable_train_critic: bool = True
    batch_size: int = 50
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

    # action ε-greedy
    epsilon: float = 0.5  # type: ignore , type OK
    test_epsilon: float = 0.0

    # 経験取得方法
    experience_acquisition_method: str = "episode_steps"  # "episode" or "loop" or "episode_steps"

    # other
    clip_rewards: str = "none"
    dummy_state_val: float = 0.0

    def __post_init__(self):
        super().__post_init__()

        self.epsilon: SchedulerConfig = SchedulerConfig(cast(float, self.epsilon))
        self.lr_model: SchedulerConfig = SchedulerConfig(cast(float, self.lr_model))
        self.lr_critic: SchedulerConfig = SchedulerConfig(cast(float, self.lr_critic))
        self.lr_actor: SchedulerConfig = SchedulerConfig(cast(float, self.lr_actor))
        self.reinforce_rate: SchedulerConfig = SchedulerConfig(cast(float, self.reinforce_rate))
        self.entropy_rate: SchedulerConfig = SchedulerConfig(cast(float, self.entropy_rate))

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
        return "DreamerV2"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.memory_warmup_size < self.capacity
        assert self.batch_size < self.memory_warmup_size

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


# ------------------------------------------------------
# Memory
# ------------------------------------------------------
class Memory(ExperienceReplayBuffer):
    pass


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _RSSM(keras.Model):
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        act=tf.nn.elu,
        use_norm_layers: bool = True,
        vae_discrete: bool = True,
    ):
        super().__init__()

        self.vae_discrete = vae_discrete

        self.rnn_cell = kl.GRUCell(deter)

        self.obs_layers = [kl.Dense(hidden, activation=act)]
        self.img_layers1 = [kl.Dense(hidden, activation=act)]
        self.img_layers2 = [kl.Dense(hidden, activation=act)]
        if use_norm_layers:
            self.obs_layers.append(kl.LayerNormalization())
            self.img_layers1.append(kl.LayerNormalization())
            self.img_layers2.append(kl.LayerNormalization())

        if self.vae_discrete:
            self.obs_discrete_dense = kl.Dense(stoch, activation=None)
            self.img_discrete_dense = kl.Dense(stoch, activation=None)
        else:
            self.obs_mean = kl.Dense(stoch, activation=None)
            self.obs_std = kl.Dense(stoch, activation=None)
            self.img_mean = kl.Dense(stoch, activation=None)
            self.img_std = kl.Dense(stoch, activation=None)

    # @tf.function
    def img_step(
        self,
        prev_stoch,
        prev_deter,
        prev_action,
        sample: bool = True,
        training: bool = False,
        _summary: bool = False,
    ):
        x = tf.concat([prev_stoch, prev_action], -1)
        for layer in self.img_layers1:
            x = layer(x, training=training)
        x, deter = self.rnn_cell(x, [prev_deter], training=training)
        deter = deter[0]
        for layer in self.img_layers2:
            x = layer(x, training=training)
        if self.vae_discrete:
            x = self.obs_discrete_dense(x)
            if _summary:
                return deter, {"logits": x}
            dist = tfd.Independent(tfd.OneHotCategorical(x), reinterpreted_batch_ndims=1)
            prior = {"logits": x}
        else:
            mean = self.img_mean(x)
            std = self.img_std(x)
            std = tf.nn.softplus(std) + 0.1
            if _summary:
                return deter, {"mean": mean, "std": std}
            dist = tfd.MultivariateNormalDiag(mean, std)
            prior = {"mean": mean, "std": std}
        prior["stoch"] = tf.cast(dist.sample() if sample else dist.mode(), tf.float32)
        return deter, prior

    # @tf.function
    def obs_step(
        self,
        deter,
        embed,
        sample: bool = True,
        training=False,
        _summary: bool = False,
    ):
        x = tf.concat([deter, embed], -1)
        for layer in self.obs_layers:
            x = layer(x, training=training)
        if self.vae_discrete:
            x = self.img_discrete_dense(x, training=training)
            if _summary:
                return {"logits": x}
            dist = tfd.Independent(tfd.OneHotCategorical(x), reinterpreted_batch_ndims=1)
            post = {"logits": x}
        else:
            mean = self.obs_mean(x)
            std = self.obs_std(x)
            std = tf.nn.softplus(std) + 0.1
            if _summary:
                return {"mean": mean, "std": std}
            dist = tfd.MultivariateNormalDiag(mean, std)
            post = {"mean": mean, "std": std}
        post["stoch"] = tf.cast(dist.sample() if sample else dist.mode(), tf.float32)
        return post

    def get_initial_state(self, batch_size: int = 1):
        return self.rnn_cell.get_initial_state(None, batch_size, dtype=tf.float32)

    def build(self, config):
        in_stoch = np.zeros((1, config.stoch_size), dtype=np.float32)
        in_deter = self.get_initial_state()
        in_action = np.zeros((1, config.action_num), dtype=np.float32)
        in_embed = np.zeros((1, 32 * config.cnn_depth), dtype=np.float32)
        deter, prior = self.img_step(in_stoch, in_deter, in_action)
        self.obs_step(deter, in_embed)
        self.built = True

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


class _Encoder(keras.Model):
    def __init__(self, depth: int = 32, act=tf.nn.relu):
        super().__init__()

        self.conv1 = kl.Conv2D(1 * depth, 4, 2, activation=act)
        self.conv2 = kl.Conv2D(2 * depth, 4, 2, activation=act)
        self.conv3 = kl.Conv2D(4 * depth, 4, 2, activation=act)
        self.conv4 = kl.Conv2D(8 * depth, 4, 2, activation=act)
        self.hout = kl.Flatten()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.hout(x)
        return x

    def build(self, input_shape):
        self._input_shape = input_shape
        super().build((1,) + self._input_shape)

    def summary(self, name="", **kwargs):
        x = kl.Input(shape=self._input_shape)
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x), name=name)
        return model.summary(**kwargs)


class _Decoder(keras.Model):
    def __init__(self, depth: int = 32, act=tf.nn.relu, fixed_variance: bool = False):
        super().__init__()
        self.fixed_variance = fixed_variance

        self.in_layer = kl.Dense(32 * depth)
        self.reshape = kl.Reshape([1, 1, 32 * depth])
        self.c1 = kl.Conv2DTranspose(4 * depth, 5, 2, activation=act)
        self.c2 = kl.Conv2DTranspose(2 * depth, 5, 2, activation=act)
        self.c3 = kl.Conv2DTranspose(1 * depth, 6, 2, activation=act)
        self.c4_mean = kl.Conv2DTranspose(3, 6, 2)
        if not fixed_variance:
            self.c4_std = kl.Conv2DTranspose(3, 6, 2)

    def call(self, x, _summary: bool = False):
        x = self.in_layer(x)
        x = self.reshape(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x_mean = self.c4_mean(x)
        if _summary:
            return x_mean
        if self.fixed_variance:
            dist = tfd.Normal(x_mean, 1)
        else:
            x_std = self.c4_std(x)
            x_std = tf.nn.softplus(x_std) + 0.1
            dist = tfd.Normal(x_mean, x_std)
        return tfd.Independent(dist, reinterpreted_batch_ndims=len(x.shape) - 1)

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build((1,) + self.__input_shape)

    def summary(self, name="", **kwargs):
        x = kl.Input(shape=self.__input_shape)
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x, _summary=True), name=name)
        return model.summary(**kwargs)


class _NormalDecoder(keras.Model):
    def __init__(
        self,
        out_shape,
        layer_sizes: Tuple[int, ...],
        act=tf.nn.elu,
        fixed_variance: bool = False,
    ):
        super().__init__()
        self._out_shape = out_shape
        self._fixed_variance = fixed_variance

        self.h_layers = [kl.Dense(units, activation=act) for units in layer_sizes]
        self.out_mean = kl.Dense(np.prod(self._out_shape))
        if not fixed_variance:
            self.out_std = kl.Dense(np.prod(self._out_shape))

    def call(self, x, _summary: bool = False):
        for layer in self.h_layers:
            x = layer(x)

        x_mean = self.out_mean(x)
        x_mean = tf.reshape(x_mean, (-1,) + self._out_shape)
        if _summary:
            return x_mean
        if self._fixed_variance:
            dist = tfd.Normal(x_mean, 1)
        else:
            x_std = self.out_std(x)
            x_std = tf.nn.softplus(x_std) + 0.1
            x_std = tf.reshape(x_std, (-1,) + self._out_shape)
            dist = tfd.Normal(x_mean, x_std)
        return tfd.Independent(dist, reinterpreted_batch_ndims=len(self._out_shape))

    def build(self, input_shape):
        self._input_shape = input_shape
        super().build((1,) + self._input_shape)

    def summary(self, name="", **kwargs):
        x = kl.Input(shape=self._input_shape)
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x, _summary=True), name=name)
        return model.summary(**kwargs)


class _BernoulliDecoder(keras.Model):
    def __init__(
        self,
        out_shape,
        layer_sizes: Tuple[int, ...],
        act=tf.nn.elu,
    ):
        super().__init__()
        self._out_shape = out_shape

        self.h_layers = [kl.Dense(units, activation=act) for units in layer_sizes]
        self.out_layer = kl.Dense(np.prod(self._out_shape))

    def call(self, x, _summary: bool = False):
        for layer in self.h_layers:
            x = layer(x)
        x = self.out_layer(x)
        x = tf.reshape(x, (-1,) + self._out_shape)
        if _summary:
            return x
        return tfd.Independent(
            tfd.Bernoulli(logits=x, dtype=tf.float32),
            reinterpreted_batch_ndims=len(self._out_shape),
        )

    def build(self, input_shape):
        self._input_shape = input_shape
        super().build((1,) + self._input_shape)

    def summary(self, name="", **kwargs):
        x = kl.Input(shape=self._input_shape)
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x, _summary=True), name=name)
        return model.summary(**kwargs)


class _OneHotDist:
    def __init__(self, logits=None, probs=None):
        self._dist = tfd.Categorical(logits=logits, probs=probs)
        self._num_classes = self.mean().shape[-1]

    @property
    def name(self):
        return "OneHotDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def prob(self, events):
        indices = tf.argmax(events, axis=-1)
        return self._dist.prob(indices)

    def log_prob(self, events):
        indices = tf.argmax(events, axis=-1)
        return self._dist.log_prob(indices)

    def mean(self):
        return self._dist.probs_parameter()

    def mode(self):
        return self._one_hot(self._dist.mode())

    def sample(self, amount=None):
        amount = [amount] if amount else []
        indices = self._dist.sample(*amount)
        sample = self._one_hot(indices)
        probs = self._dist.probs_parameter()
        sample += probs - tf.stop_gradient(probs)
        return sample

    def _one_hot(self, indices):
        return tf.one_hot(indices, self._num_classes, dtype=tf.float32)


class _TanhBijector(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name="tanh"):
        super().__init__(forward_min_event_ndims=0, validate_args=validate_args, name=name)

    def _forward(self, x):
        return tf.nn.tanh(x)

    def _inverse(self, y):
        dtype = y.dtype
        y = tf.cast(y, tf.float32)
        y = tf.where(tf.less_equal(tf.abs(y), 1.0), tf.clip_by_value(y, -0.99999997, 0.99999997), y)
        y = tf.atanh(y)
        y = tf.cast(y, dtype)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = tf.math.log(tf.constant(2.0, dtype=x.dtype))
        return 2.0 * (log2 - x - tf.nn.softplus(-2.0 * x))


class _SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return tf.reduce_mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return tf.gather(sample, tf.argmax(logprob))[0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -tf.reduce_mean(logprob, 0)


class _ActorDiscreteDecoder(keras.Model):
    def __init__(
        self,
        action_num: int,
        layer_sizes: Tuple[int, ...],
    ):
        super().__init__()
        self.dense_layers = [kl.Dense(units, activation="elu") for units in layer_sizes]
        self.out = kl.Dense(action_num)

    def call(self, x, return_logits: bool = False):
        for layer in self.dense_layers:
            x = layer(x)
        x = self.out(x)
        dist = _OneHotDist(logits=x)
        if return_logits:
            return dist, x
        else:
            return dist

    def build(self, input_shape):
        self._input_shape = input_shape
        super().build((1,) + self._input_shape)

    def summary(self, name="", **kwargs):
        x = in_x = kl.Input(shape=self._input_shape)
        for layer in self.dense_layers:
            x = layer(x)
        x = self.out(x)
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=in_x, outputs=x, name=name)
        return model.summary(**kwargs)


class _ActorContinuousDecoder(keras.Model):
    def __init__(
        self,
        action_num: int,
        layer_sizes: Tuple[int, ...],
        min_std=1e-4,
        init_std=5.0,
        mean_scale=5.0,
    ):
        super().__init__()
        self._min_std = min_std
        self._mean_scale = mean_scale
        self._raw_init_std = np.log(np.exp(init_std) - 1)

        self.dense_layers = [kl.Dense(units, activation="elu") for units in layer_sizes]
        self.out = kl.Dense(action_num * 2)

    def call(self, x, return_logits: bool = False):
        for layer in self.dense_layers:
            x = layer(x)

        # tanh_normal
        x = self.out(x)
        mean, std = tf.split(x, 2, -1)
        mean = self._mean_scale * tf.tanh(mean / self._mean_scale)
        std = tf.nn.softplus(std + self._raw_init_std) + self._min_std
        dist = tfd.Normal(mean, std)
        dist = tfd.TransformedDistribution(dist, _TanhBijector())
        dist = tfd.Independent(dist, 1)
        dist = _SampleDist(dist)
        if return_logits:
            return dist, x
        else:
            return dist

    def build(self, input_shape):
        self._input_shape = input_shape
        super().build((1,) + self._input_shape)

    def summary(self, name="", **kwargs):
        x = in_x = kl.Input(shape=self._input_shape)
        for layer in self.dense_layers:
            x = layer(x)
        x = self.out(x)
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=in_x, outputs=x, name=name)
        return model.summary(**kwargs)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.encode = _Encoder(self.config.cnn_depth, self.config.cnn_act)
        self.dynamics = _RSSM(self.config.stoch_size, self.config.deter_size, self.config.deter_size)
        self.decode = _Decoder(self.config.cnn_depth, self.config.cnn_act)
        self.reward = _NormalDecoder(
            (1,),
            self.config.reward_layer_sizes,
            self.config.dense_act,
            self.config.fixed_variance,
        )
        self.discount = _BernoulliDecoder(
            (1,),
            self.config.discount_layer_sizes,
            self.config.dense_act,
        )
        self.critic = _NormalDecoder(
            (1,),
            self.config.critic_layer_sizes,
            self.config.dense_act,
            self.config.fixed_variance,
        )
        self.critic_target = _NormalDecoder(
            (1,),
            self.config.critic_layer_sizes,
            self.config.dense_act,
            self.config.fixed_variance,
        )
        self.actor = _ActorDiscreteDecoder(
            self.config.action_num,
            self.config.actor_layer_sizes,
        )

        self.encode.build(self.config.observation_shape)
        self.dynamics.build(self.config)
        self.decode.build((self.config.deter_size + self.config.stoch_size,))
        self.reward.build((self.config.deter_size + self.config.stoch_size,))
        self.discount.build((self.config.deter_size + self.config.stoch_size,))
        self.critic.build((self.config.deter_size + self.config.stoch_size,))
        self.critic_target.build((self.config.deter_size + self.config.stoch_size,))
        self.actor.build((self.config.deter_size + self.config.stoch_size,))

        self.critic_target.set_weights(self.critic.get_weights())

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

        self.lr_sch_model = self.config.lr_model.create_schedulers()
        self.lr_sch_critic = self.config.lr_critic.create_schedulers()
        self.lr_sch_actor = self.config.lr_actor.create_schedulers()
        self.reinforce_rate_sch = self.config.reinforce_rate.create_schedulers()
        self.entropy_rate_sch = self.config.entropy_rate.create_schedulers()

        if compare_less_version(tf.__version__, "2.11.0"):
            self._model_opt = keras.optimizers.Adam(learning_rate=self.lr_sch_model.get_rate(0))
            self._critic_opt = keras.optimizers.Adam(learning_rate=self.lr_sch_critic.get_rate(0))
            self._actor_opt = keras.optimizers.Adam(learning_rate=self.lr_sch_actor.get_rate(0))
        else:
            self._model_opt = keras.optimizers.legacy.Adam(learning_rate=self.lr_sch_model.get_rate(0))
            self._critic_opt = keras.optimizers.legacy.Adam(learning_rate=self.lr_sch_critic.get_rate(0))
            self._actor_opt = keras.optimizers.legacy.Adam(learning_rate=self.lr_sch_actor.get_rate(0))

        self.sync_count = 0

    def train_on_batchs(self, memory_sample_return) -> None:
        batchs = memory_sample_return
        info = {}

        states = np.asarray([b["states"] for b in batchs], dtype=np.float32)
        actions = [b["actions"] for b in batchs]
        rewards = np.asarray([b["rewards"] for b in batchs], dtype=np.float32)[..., np.newaxis]
        discounts = np.asarray([b["discounts"] for b in batchs], dtype=np.float32)[..., np.newaxis]

        # onehot action
        actions = tf.one_hot(actions, self.config.action_num, axis=2)

        # (batch, seq, shape) -> (batch * seq, shape)
        states = tf.reshape(states, (self.config.batch_size * self.config.batch_length,) + states.shape[2:])
        rewards = tf.reshape(rewards, (self.config.batch_size * self.config.batch_length,) + rewards.shape[2:])
        discounts = tf.reshape(discounts, (self.config.batch_size * self.config.batch_length,) + discounts.shape[2:])

        # ------------------------
        # RSSM
        # ------------------------
        self.parameter.encode.trainable = True
        self.parameter.decode.trainable = True
        self.parameter.dynamics.trainable = True
        self.parameter.reward.trainable = True
        self.parameter.actor.trainable = False
        self.parameter.critic.trainable = False
        with tf.GradientTape() as tape:
            embed = self.parameter.encode(states, training=True)
            embed_shape = embed.shape

            # (batch * seq, shape) -> (batch, seq, shape)
            # (batch, seq, shape) -> (seq, batch, shape)
            shape = (self.config.batch_size, self.config.batch_length) + embed_shape[1:]
            embed = tf.reshape(embed, shape)
            embed = tf.transpose(embed, [1, 0, 2])
            actions = tf.transpose(actions, [1, 0, 2])

            stochs = []
            deters = []
            stoch = tf.zeros([self.config.batch_size, self.config.stoch_size], dtype=tf.float32)
            deter = self.parameter.dynamics.get_initial_state(self.config.batch_size)
            if self.config.vae_discrete:
                post_logits = []
                prior_logits = []
            else:
                post_mean = []
                post_std = []
                prior_mean = []
                prior_std = []
            for i in range(self.config.batch_length):
                deter, prior = self.parameter.dynamics.img_step(stoch, deter, actions[i], training=True)
                post = self.parameter.dynamics.obs_step(deter, embed[i], training=True)
                stochs.append(post["stoch"])
                deters.append(deter)
                if self.config.vae_discrete:
                    post_logits.append(post["logits"])
                    prior_logits.append(prior["logits"])
                else:
                    post_mean.append(post["mean"])
                    post_std.append(post["std"])
                    prior_mean.append(prior["mean"])
                    prior_std.append(prior["std"])
            stochs = tf.stack(stochs, axis=0)
            deters = tf.stack(deters, axis=0)

            # (seq, batch, shape) -> (batch, seq, shape)
            stochs = tf.transpose(stochs, [1, 0, 2])
            deters = tf.transpose(deters, [1, 0, 2])

            feat = tf.concat([stochs, deters], -1)
            feat = tf.reshape(feat, (self.config.batch_size * self.config.batch_length,) + feat.shape[2:])
            image_pred = self.parameter.decode(feat)
            reward_pred = self.parameter.reward(feat)
            discount_pred = self.parameter.discount(feat)

            image_loss = image_pred.log_prob(states)
            reward_loss = reward_pred.log_prob(rewards)
            discount_loss = discount_pred.log_prob(discounts)

            if self.config.vae_discrete:
                post_logits = tf.stack(post_logits, axis=0)
                prior_logits = tf.stack(prior_logits, axis=0)

                # (seq, batch, shape) -> (batch, seq, shape)
                post_logits = tf.transpose(post_logits, [1, 0, 2])
                prior_logits = tf.transpose(prior_logits, [1, 0, 2])

                prior_dist = tfd.Independent(tfd.OneHotCategorical(post_logits), reinterpreted_batch_ndims=1)
                post_dist = tfd.Independent(tfd.OneHotCategorical(prior_logits), reinterpreted_batch_ndims=1)

            else:
                post_mean = tf.stack(post_mean, axis=0)
                post_std = tf.stack(post_std, axis=0)
                prior_mean = tf.stack(prior_mean, axis=0)
                prior_std = tf.stack(prior_std, axis=0)

                # (seq, batch, shape) -> (batch, seq, shape)
                post_mean = tf.transpose(post_mean, [1, 0, 2])
                post_std = tf.transpose(post_std, [1, 0, 2])
                prior_mean = tf.transpose(prior_mean, [1, 0, 2])
                prior_std = tf.transpose(prior_std, [1, 0, 2])

                prior_dist = tfd.MultivariateNormalDiag(prior_mean, prior_std)
                post_dist = tfd.MultivariateNormalDiag(post_mean, post_std)

            kl_loss = self.config.kl_balancing_rate * tfd.kl_divergence(tf.stop_gradient(post_dist), prior_dist)
            kl_loss += (1 - self.config.kl_balancing_rate) * tfd.kl_divergence(post_dist, tf.stop_gradient(prior_dist))
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss = tf.maximum(kl_loss, self.config.free_nats)
            loss = self.config.kl_scale * kl_loss - tf.reduce_mean(image_loss + reward_loss + discount_loss)

        if self.config.enable_train_model:
            variables = [
                self.parameter.encode.trainable_variables,
                self.parameter.dynamics.trainable_variables,
                self.parameter.decode.trainable_variables,
                self.parameter.reward.trainable_variables,
            ]
            grads = tape.gradient(loss, variables)
            for i in range(len(variables)):
                self._model_opt.apply_gradients(zip(grads[i], variables[i]))

            lr = self.lr_sch_model.get_rate(self.train_count)
            self._model_opt.learning_rate = lr

            info["img_loss"] = -np.mean(image_loss.numpy()) / (64 * 64 * 3)
            info["reward_loss"] = -np.mean(reward_loss.numpy())
            info["discount_loss"] = -np.mean(discount_loss.numpy())
            info["kl_loss"] = kl_loss.numpy()
            info["model_lr"] = lr

        if (not self.config.enable_train_actor) and (not self.config.enable_train_critic):
            # WorldModelsのみ学習
            self.train_count += 1
            self.train_info = info
            return

        self.parameter.encode.trainable = False
        self.parameter.decode.trainable = False
        self.parameter.dynamics.trainable = False
        self.parameter.reward.trainable = False

        # batch + 各step を最初として、 初期状態を作成
        stochs = tf.reshape(stochs, (self.config.batch_size * self.config.batch_length,) + stochs.shape[2:])
        deters = tf.reshape(deters, (self.config.batch_size * self.config.batch_length,) + deters.shape[2:])
        feats = tf.concat([stochs, deters], -1)

        # ------------------------
        # Actor
        # ------------------------
        horizon_feats = None
        if self.config.enable_train_actor:
            self.parameter.actor.trainable = True
            self.parameter.critic.trainable = False
            reinforce_rate = self.reinforce_rate_sch.get_rate(self.train_count)
            entropy_rate = self.entropy_rate_sch.get_rate(self.train_count)
            with tf.GradientTape() as tape:
                horizon_feats, horizon_log_pi, horizon_v, horizon_V = self._compute_horizon_step(stochs, deters, feats)
                # (horizon, batch_size*batch_length, 1)

                # reinforce
                if self.config.reinforce_baseline == "v":
                    adv = tf.stop_gradient(horizon_V - horizon_v)
                else:
                    adv = tf.stop_gradient(horizon_V)
                reinforce_loss = -horizon_log_pi * adv

                # dynamics backprop
                dynamics_loss = -horizon_V

                # entropy
                entropy_loss = -horizon_log_pi

                act_loss = tf.reduce_mean(
                    reinforce_rate * reinforce_loss
                    + (1 - reinforce_rate) * dynamics_loss
                    + entropy_rate * entropy_loss
                )

            grads = tape.gradient(act_loss, self.parameter.actor.trainable_variables)
            self._actor_opt.apply_gradients(zip(grads, self.parameter.actor.trainable_variables))
            info["reinforce_loss"] = np.mean(reinforce_loss)
            info["dynamics_loss"] = np.mean(dynamics_loss)
            info["entropy_loss"] = np.mean(entropy_loss)
            info["act_loss"] = act_loss.numpy()

            lr = self.lr_sch_actor.get_rate(self.train_count)
            self._actor_opt.learning_rate = lr
            info["act_lr"] = lr

        # ------------------------
        # critic
        # ------------------------
        if self.config.enable_train_critic:
            if horizon_feats is None:
                horizon_feats, horizon_log_pi, horizon_v, horizon_V = self._compute_horizon_step(stochs, deters, feats)

            # (horizon, batch_size*batch_length, feat) -> (horizon*batch_size*batch_length, feat)
            horizon_feats = tf.reshape(
                horizon_feats,
                (horizon_feats.shape[0] * horizon_feats.shape[1], horizon_feats.shape[2]),
            )
            horizon_V = tf.reshape(
                horizon_V,
                (horizon_V.shape[0] * horizon_V.shape[1], horizon_V.shape[2]),
            )

            self.parameter.actor.trainable = False
            self.parameter.critic.trainable = True
            with tf.GradientTape() as tape:
                critic_pred = self.parameter.critic(horizon_feats)
                critic_loss = -tf.reduce_mean(critic_pred.log_prob(horizon_V))
            grads = tape.gradient(critic_loss, self.parameter.critic.trainable_variables)
            self._critic_opt.apply_gradients(zip(grads, self.parameter.critic.trainable_variables))
            info["critic_loss"] = critic_loss.numpy()

            lr = self.lr_sch_critic.get_rate(self.train_count)
            self._critic_opt.learning_rate = lr
            info["critic_lr"] = lr

            # --- targetと同期
            if self.train_count % self.config.target_critic_update_interval == 0:
                self.parameter.critic_target.set_weights(self.parameter.critic.get_weights())
                self.sync_count += 1
            info["critic_sync"] = self.sync_count

        self.train_count += 1
        self.train_info = info

    def _compute_horizon_step(
        self,
        stochs,
        deters,
        feats,
    ):
        horizon_feats = []
        horizon_log_pi = []
        horizon_v = []

        if self.config.critic_estimation_method == "simple":
            _horizon_reward = []
            for t in range(self.config.horizon):
                stochs, deters, feats, log_pi = self._horizon_step(stochs, deters, feats)
                horizon_feats.append(feats)
                horizon_log_pi.append(log_pi)
                horizon_v.append(self.parameter.critic_target(feats).mode())
                _horizon_reward.append(self.parameter.reward(feats).mode())

            # 累積和の平均
            horizon_V = tf.math.cumsum(_horizon_reward, reverse=True)
            weights = tf.reshape(
                1.0 / tf.range(len(horizon_V), 0, -1, dtype=tf.float32),
                (len(horizon_V), 1, 1),
            )
            weights = tf.tile(weights, (1, horizon_V.shape[1], horizon_V.shape[2]))
            horizon_V *= weights

        elif self.config.critic_estimation_method == "dreamer":
            _horizon_V = []
            for t in range(self.config.horizon):
                stochs, deters, feats, log_pi = self._horizon_step(stochs, deters, feats)
                horizon_feats.append(feats)
                horizon_log_pi.append(log_pi)
                reward = self.parameter.reward(feats).mode()
                v = self.parameter.critic_target(feats).mode()
                discount = self.parameter.discount(feats).mode()
                horizon_v.append(v)
                _horizon_V.append((discount**t) * reward + (discount ** (self.config.horizon - t)) * v)

            _horizon_V = tf.math.cumsum(_horizon_V, reverse=True)
            weights = tf.reshape(
                1.0 / tf.range(len(_horizon_V), 0, -1, dtype=tf.float32),
                (len(_horizon_V), 1, 1),
            )
            weights = tf.tile(weights, (1, _horizon_V.shape[1], _horizon_V.shape[2]))
            _horizon_V *= weights

            # EWA
            v = (1 - self.config.disclam) * _horizon_V[0]
            horizon_V = [v]
            for t in range(1, self.config.horizon):
                v = self.config.disclam * v + (1 - self.config.disclam) * _horizon_V[t]
                horizon_V.append(v)
            horizon_V = tf.stack(horizon_V)

        elif self.config.critic_estimation_method == "dreamer_v2":
            _horizons = []
            for t in range(self.config.horizon):
                stochs, deters, feats, log_pi = self._horizon_step(stochs, deters, feats)
                horizon_feats.append(feats)
                horizon_log_pi.append(log_pi)
                reward = self.parameter.reward(feats).mode()
                discount = self.parameter.discount(feats).mean()
                v = self.parameter.critic_target(feats).mode()
                horizon_v.append(v)
                _horizons.append([reward, discount, v])

            horizon_V = []
            V = None
            for reward, discount, v in reversed(_horizons):
                if V is None:
                    V = reward + discount * v
                else:
                    gain = (1 - self.config.h_target) * v + self.config.h_target * V
                    V = reward + discount * gain
                horizon_V.append(V)
            horizon_V = tf.stack(horizon_V)

            weights = tf.reshape(
                1.0 / tf.range(len(horizon_V), 0, -1, dtype=tf.float32),
                (len(horizon_V), 1, 1),
            )
            weights = tf.tile(weights, (1, horizon_V.shape[1], horizon_V.shape[2]))
            horizon_V *= weights

        else:
            raise ValueError(self.config.critic_estimation_method)

        horizon_feats = tf.stack(horizon_feats)
        horizon_log_pi = tf.expand_dims(tf.stack(horizon_log_pi), axis=-1)
        horizon_v = tf.stack(horizon_v)
        return horizon_feats, horizon_log_pi, horizon_v, horizon_V

    def _horizon_step(self, stoch, deter, feat):
        dist = self.parameter.actor(feat)
        action = dist.sample()
        log_pi = dist.log_prob(action)
        deter, prior = self.parameter.dynamics.img_step(stoch, deter, action)
        stoch = prior["stoch"]
        feat = tf.concat([stoch, deter], -1)

        return stoch, deter, feat, log_pi


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

        self.epsilon_sch = self.config.epsilon.create_schedulers()

        self._recent_states = []
        self._recent_actions = []
        self._recent_rewards = []
        self._recent_discounts = []

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        if self.config.experience_acquisition_method in ["episode", "episode_steps"]:
            self._recent_states = []
            self._recent_actions = []
            self._recent_rewards = []
            self._recent_discounts = []

        self.deter = self.parameter.dynamics.get_initial_state()
        self.stoch = tf.zeros((1, self.config.stoch_size), dtype=tf.float32)
        self.action = 0
        self.prev_action = 0

        return {}

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        self.invalid_actions = invalid_actions
        self.state = state
        self.feat = None
        self.prev_action = self.action

        if self.training:
            epsilon = self.epsilon_sch.get_rate(self.total_step)
        else:
            epsilon = self.config.test_epsilon

        if epsilon < 1.0:
            # 少しでもactorでaction決定する可能性があれば、rssmを進める
            self._rssm_step()

        if random.random() < epsilon:
            self.action = cast(int, self.sample_action())
            return self.action, {}

        action = self.parameter.actor(self.feat).mode()
        self.action = np.argmax(action[0])

        return int(self.action), {}

    def _rssm_step(self):
        embed = self.parameter.encode(self.state[np.newaxis, ...])
        prev_action = tf.one_hot([self.prev_action], self.config.action_num, axis=1)
        deter, prior = self.parameter.dynamics.img_step(self.stoch, self.deter, prev_action, sample=self.training)
        post = self.parameter.dynamics.obs_step(deter, embed, sample=self.training)
        self.feat = tf.concat([post["stoch"], deter], axis=1)
        self.deter = deter
        self.stoch = post["stoch"]

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ) -> dict:
        if not self.training:
            return {}

        clip_rewards_fn = dict(none=lambda x: x, tanh=tf.tanh)[self.config.clip_rewards]
        reward = clip_rewards_fn(reward)

        if self.config.experience_acquisition_method == "loop":
            # エピソードをまたいで一定stepでバッチとする
            self._recent_states.append(next_state)
            self._recent_actions.append(self.action)
            self._recent_rewards.append(reward)
            self._recent_discounts.append(0 if done else self.config.discount)
            if len(self._recent_states) == self.config.batch_length:
                self.memory.add(
                    {
                        "states": self._recent_states,
                        "actions": self._recent_actions,
                        "rewards": self._recent_rewards,
                        "discounts": self._recent_discounts,
                    }
                )
                self._recent_states = []
                self._recent_actions = []
                self._recent_rewards = []
                self._recent_discounts = []
        elif self.config.experience_acquisition_method == "episode":
            # 1エピソードの0stepからbatch_lengthをバッチとする
            # batch_length以降のstepは無視
            if len(self._recent_states) < self.config.batch_length:
                self._recent_states.append(next_state)
                self._recent_actions.append(self.action)
                self._recent_rewards.append(reward)
                self._recent_discounts.append(0 if done else self.config.discount)

            if done:
                # 足りない分はダミーデータで補完
                for _ in range(self.config.batch_length - len(self._recent_states)):
                    self._recent_states.append(next_state)
                    self._recent_actions.append(random.randint(0, self.config.action_num - 1))
                    self._recent_rewards.append(reward)
                    self._recent_discounts.append(0)

                self.memory.add(
                    {
                        "states": self._recent_states,
                        "actions": self._recent_actions,
                        "rewards": self._recent_rewards,
                        "discounts": self._recent_discounts,
                    }
                )

        else:
            # 1エピソードにて、毎stepからbatch_length分をバッチとする
            self._recent_states.append(next_state)
            self._recent_actions.append(self.action)
            self._recent_rewards.append(reward)
            self._recent_discounts.append(0 if done else self.config.discount)

            if done:
                for i in range(len(self._recent_rewards)):
                    batch_states = self._recent_states[i : i + self.config.batch_length]
                    batch_actions = self._recent_actions[i : i + self.config.batch_length]
                    batch_rewards = self._recent_rewards[i : i + self.config.batch_length]
                    batch_discounts = self._recent_discounts[i : i + self.config.batch_length]

                    # 足りない分はダミーデータで補完
                    for _ in range(self.config.batch_length - len(batch_rewards)):
                        batch_states.append(next_state)
                        batch_actions.append(random.randint(0, self.config.action_num - 1))
                        batch_rewards.append(reward)
                        batch_discounts.append(0)

                    self.memory.add(
                        {
                            "states": batch_states,
                            "actions": batch_actions,
                            "rewards": batch_rewards,
                            "discounts": batch_discounts,
                        }
                    )

        return {}

    def render_terminal(self, env, worker, **kwargs) -> None:
        pass

    def render_rgb_array(self, env, worker, **kwargs) -> Optional[np.ndarray]:
        if self.config.env_observation_type != EnvObservationTypes.COLOR:
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
                f"{env.action_to_str(a)}({policy_logits[a]:.2f})",
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
