import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from srl.base.define import EnvObservationTypes, RLTypes
from srl.base.rl.algorithms.discrete_action import DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.rl.registration import register
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, ExperienceReplayBufferConfig
from srl.rl.processors.image_processor import ImageProcessor
from srl.utils.common import compare_less_version

kl = keras.layers
tfd = tfp.distributions

"""
paper: https://arxiv.org/abs/1912.01603
ref: https://github.com/danijar/dreamer
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig, ExperienceReplayBufferConfig):
    capacity: int = 100_000
    memory_warmup_size: int = 1000

    # Model
    deter_size: int = 200
    stoch_size: int = 30
    reward_num_units: int = 400
    reward_layers: int = 2
    value_num_units: int = 400
    value_layers: int = 3
    action_num_units: int = 400
    action_layers: int = 4
    dense_act: Any = "elu"
    cnn_act: Any = "relu"
    cnn_depth: int = 32
    free_nats: float = 3.0
    kl_scale: float = 1.0

    # Training
    enable_train_model: bool = True
    enable_train_actor: bool = True
    enable_train_value: bool = True
    batch_size: int = 50
    batch_length: int = 50
    model_lr: float = 6e-4
    value_lr: float = 8e-5
    actor_lr: float = 8e-5

    # Behavior
    discount: float = 0.99
    disclam: float = 0.95
    horizon: int = 15
    value_estimation_method: str = "dreamer"  # "simple" or "dreamer"

    # action ε-greedy
    epsilon: float = 0.5
    test_epsilon: float = 0.0

    # 経験取得方法
    experience_acquisition_method: str = "episode"  # "episode" or "loop"

    # other
    clip_rewards: str = "none"
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
        return "Dreamer"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.memory_warmup_size < self.capacity
        assert self.batch_size < self.memory_warmup_size


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
class RemoteMemory(ExperienceReplayBuffer):
    pass


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _RSSM(keras.Model):
    def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
        super().__init__()

        self.rnn_cell = kl.GRUCell(deter)
        self.obs1 = kl.Dense(hidden, activation=act)
        self.obs_mean = kl.Dense(stoch, activation=None)
        self.obs_std = kl.Dense(stoch, activation=None)
        self.img1 = kl.Dense(hidden, activation=act)
        self.img2 = kl.Dense(hidden, activation=act)
        self.img_mean = kl.Dense(stoch, activation=None)
        self.img_std = kl.Dense(stoch, activation=None)

    def obs_step(self, prev_stoch, prev_deter, prev_action, embed, training=False, _summary: bool = False):
        deter, prior = self.img_step(prev_stoch, prev_deter, prev_action, training=training, _summary=_summary)
        x = tf.concat([deter, embed], -1)
        x = self.obs1(x)
        mean = self.obs_mean(x)
        std = self.obs_std(x)
        std = tf.nn.softplus(std) + 0.1
        if _summary:
            return [mean, std, prior["mean"], prior["std"]]
        stoch = tfd.MultivariateNormalDiag(mean, std).sample()
        post = {"mean": mean, "std": std, "stoch": stoch}
        return post, deter, prior

    def img_step(self, prev_stoch, prev_deter, prev_action, training=False, _summary: bool = False):
        x = tf.concat([prev_stoch, prev_action], -1)
        x = self.img1(x)
        x, deter = self.rnn_cell(x, [prev_deter], training=training)  # type:ignore , ignore check "None"
        deter = deter[0]
        x = self.img2(x)
        mean = self.img_mean(x)
        std = self.img_std(x)
        std = tf.nn.softplus(std) + 0.1
        if _summary:
            return deter, {"mean": mean, "std": std}
        stoch = tfd.MultivariateNormalDiag(mean, std).sample()
        prior = {"mean": mean, "std": std, "stoch": stoch}
        return deter, prior

    def get_initial_state(self, batch_size: int = 1):
        return self.rnn_cell.get_initial_state(None, batch_size, dtype=tf.float32)

    def build(self, config):
        in_stoch = np.zeros((1, config.stoch_size), dtype=np.float32)
        in_deter = self.get_initial_state()
        in_action = np.zeros((1, config.action_num), dtype=np.float32)
        in_embed = np.zeros((1, 32 * config.cnn_depth), dtype=np.float32)
        self.obs_step(in_stoch, in_deter, in_action, in_embed)
        self.built = True

    def summary(self, config, **kwargs):
        in_stoch = kl.Input((config.stoch_size,))
        in_deter = kl.Input((config.deter_size,))
        in_action = kl.Input((config.action_num,))
        in_embed = kl.Input((32 * config.cnn_depth,))

        model = keras.Model(
            inputs=[in_stoch, in_deter, in_action, in_embed],
            outputs=self.obs_step(in_stoch, in_deter, in_action, in_embed, _summary=True),
            name="RSSM",
        )
        return model.summary(**kwargs)


class _ConvEncoder(keras.Model):
    def __init__(self, depth: int = 32, act=tf.nn.relu):
        super().__init__()

        kwargs = dict(kernel_size=4, strides=2, activation=act)
        self.conv1 = kl.Conv2D(filters=1 * depth, **kwargs)
        self.conv2 = kl.Conv2D(filters=2 * depth, **kwargs)
        self.conv3 = kl.Conv2D(filters=4 * depth, **kwargs)
        self.conv4 = kl.Conv2D(filters=8 * depth, **kwargs)
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


class _ConvDecoder(keras.Model):
    def __init__(self, depth: int = 32, act=tf.nn.relu):
        super().__init__()

        kwargs = dict(strides=2, activation=act)
        self.in_layer = kl.Dense(32 * depth)
        self.reshape = kl.Reshape([1, 1, 32 * depth])
        self.c1 = kl.Conv2DTranspose(4 * depth, kernel_size=5, **kwargs)
        self.c2 = kl.Conv2DTranspose(2 * depth, kernel_size=5, **kwargs)
        self.c3 = kl.Conv2DTranspose(1 * depth, kernel_size=6, **kwargs)
        self.c4_mean = kl.Conv2DTranspose(3, kernel_size=6, strides=2)
        self.c4_std = kl.Conv2DTranspose(3, kernel_size=6, strides=2)

    def call(self, x, _summary: bool = False):
        x = self.in_layer(x)
        x = self.reshape(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x_mean = self.c4_mean(x)
        x_std = self.c4_std(x)
        x_std = tf.nn.softplus(x_std) + 0.1
        if _summary:
            return x_mean
        return tfd.Independent(
            tfd.Normal(x_mean, x_std),
            reinterpreted_batch_ndims=len(x.shape) - 1,  # type:ignore , ignore check "None"
        )

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build((1,) + self.__input_shape)

    def summary(self, name="", **kwargs):
        x = kl.Input(shape=self.__input_shape)
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x, _summary=True), name=name)
        return model.summary(**kwargs)


class _DenseDecoder(keras.Model):
    def __init__(self, out_shape, layers: int, units: int, dist: str = "normal", act=tf.nn.elu):
        super().__init__()
        self._out_shape = out_shape
        self._dist = dist

        self.h_layers = [kl.Dense(units, activation=act) for i in range(layers)]
        self.hout_mean = kl.Dense(np.prod(self._out_shape))
        self.hout_std = kl.Dense(np.prod(self._out_shape))

    def call(self, x, _summary: bool = False):
        for layer in self.h_layers:
            x = layer(x)
        x_mean = self.hout_mean(x)
        x_std = self.hout_std(x)
        x_std = tf.nn.softplus(x_std) + 0.1
        x_mean = tf.reshape(x_mean, (-1,) + self._out_shape)
        x_std = tf.reshape(x_std, (-1,) + self._out_shape)
        if _summary:
            return x_mean
        if self._dist == "normal":
            return tfd.Independent(tfd.Normal(x_mean, x_std), reinterpreted_batch_ndims=len(self._out_shape))
        if self._dist == "binary":
            return tfd.Independent(tfd.Bernoulli(x), reinterpreted_batch_ndims=len(self._out_shape))
        raise NotImplementedError(self._dist)

    def build(self, input_shape):
        self._input_shape = input_shape
        super().build((1,) + self._input_shape)

    def summary(self, name="", **kwargs):
        x = kl.Input(shape=self._input_shape)
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x, _summary=True), name=name)
        return model.summary(**kwargs)


class OneHotDist:
    def __init__(self, logits=None, probs=None):
        self._dist = tfd.Categorical(logits=logits, probs=probs)
        self._num_classes = self.mean().shape[-1]
        self._dtype = tf.float32

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
        sample += tf.cast(probs - tf.stop_gradient(probs), self._dtype)
        return sample

    def _one_hot(self, indices):
        return tf.one_hot(indices, self._num_classes, dtype=self._dtype)


class TanhBijector(tfp.bijectors.Bijector):
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


class SampleDist:
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


class _ActionDiscreteDecoder(keras.Model):
    def __init__(
        self,
        action_num: int,
        layers: int,
        units: int,
    ):
        super().__init__()
        self.dense_layers = [kl.Dense(units, activation="elu") for i in range(layers)]
        self.out = kl.Dense(action_num)

    def call(self, x, return_logits: bool = False):
        for layer in self.dense_layers:
            x = layer(x)
        x = self.out(x)
        dist = OneHotDist(logits=x)
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


class _ActionContinuousDecoder(keras.Model):
    def __init__(
        self,
        action_num: int,
        layers: int,
        units: int,
        min_std=1e-4,
        init_std=5.0,
        mean_scale=5.0,
    ):
        super().__init__()
        self._min_std = min_std
        self._mean_scale = mean_scale
        self._raw_init_std = np.log(np.exp(init_std) - 1)

        self.dense_layers = [kl.Dense(units, activation="elu") for i in range(layers)]
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
        dist = tfd.TransformedDistribution(dist, TanhBijector())
        dist = tfd.Independent(dist, 1)
        dist = SampleDist(dist)
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

        self.encode = _ConvEncoder(self.config.cnn_depth, self.config.cnn_act)
        self.dynamics = _RSSM(self.config.stoch_size, self.config.deter_size, self.config.deter_size)
        self.decode = _ConvDecoder(self.config.cnn_depth, self.config.cnn_act)
        self.reward = _DenseDecoder(
            (1,),
            self.config.reward_layers,
            self.config.reward_num_units,
            "normal",
            self.config.dense_act,
        )
        self.value = _DenseDecoder(
            (1,),
            self.config.value_layers,
            self.config.value_num_units,
            "normal",
            self.config.dense_act,
        )
        self.actor = _ActionDiscreteDecoder(
            self.config.action_num, self.config.action_layers, self.config.action_num_units
        )

        self.encode.build(self.config.observation_shape)
        self.dynamics.build(self.config)
        self.decode.build((self.config.deter_size + self.config.stoch_size,))
        self.reward.build((self.config.deter_size + self.config.stoch_size,))
        self.value.build((self.config.deter_size + self.config.stoch_size,))
        self.actor.build((self.config.deter_size + self.config.stoch_size,))

    def call_restore(self, data: Any, **kwargs) -> None:
        self.encode.set_weights(data[0])
        self.dynamics.set_weights(data[1])
        self.decode.set_weights(data[2])
        self.reward.set_weights(data[3])
        self.value.set_weights(data[4])
        self.actor.set_weights(data[5])

    def call_backup(self, **kwargs) -> Any:
        return [
            self.encode.get_weights(),
            self.dynamics.get_weights(),
            self.decode.get_weights(),
            self.reward.get_weights(),
            self.value.get_weights(),
            self.actor.get_weights(),
        ]

    def summary(self, **kwargs):
        self.encode.summary("Encoder", **kwargs)
        self.dynamics.summary(self.config, **kwargs)
        self.decode.summary("Decoder", **kwargs)
        self.reward.summary("Reward", **kwargs)
        self.value.summary("Value", **kwargs)
        self.actor.summary("Actor", **kwargs)


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter
        self.remote_memory: RemoteMemory = self.remote_memory

        if compare_less_version(tf.__version__, "2.11.0"):
            self._model_opt = keras.optimizers.Adam(learning_rate=self.config.model_lr)
            self._value_opt = keras.optimizers.Adam(learning_rate=self.config.value_lr)
            self._actor_opt = keras.optimizers.Adam(learning_rate=self.config.actor_lr)
        else:
            self._model_opt = keras.optimizers.legacy.Adam(learning_rate=self.config.model_lr)
            self._value_opt = keras.optimizers.legacy.Adam(learning_rate=self.config.value_lr)
            self._actor_opt = keras.optimizers.legacy.Adam(learning_rate=self.config.actor_lr)

        self.train_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):
        if self.remote_memory.length() < self.config.memory_warmup_size:
            return {}
        info = {}

        batchs = self.remote_memory.sample(self.config.batch_size)

        states = np.asarray([b["states"] for b in batchs], dtype=np.float32)
        actions = [b["actions"] for b in batchs]
        rewards = np.asarray([b["rewards"] for b in batchs], dtype=np.float32)[..., np.newaxis]

        # onehot action
        actions = tf.one_hot(actions, self.config.action_num, axis=2)

        # (batch, seq, shape) -> (batch * seq, shape)
        states = tf.reshape(states, (self.config.batch_size * self.config.batch_length,) + states.shape[2:])
        rewards = tf.reshape(rewards, (self.config.batch_size * self.config.batch_length,) + rewards.shape[2:])

        # ------------------------
        # RSSM
        # ------------------------
        self.parameter.encode.trainable = True
        self.parameter.decode.trainable = True
        self.parameter.dynamics.trainable = True
        self.parameter.reward.trainable = True
        self.parameter.actor.trainable = False
        self.parameter.value.trainable = False
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
            post_mean = []
            post_std = []
            prior_mean = []
            prior_std = []
            for i in range(self.config.batch_length):
                post, deter, prior = self.parameter.dynamics.obs_step(
                    stoch, deter, actions[i], embed[i], training=True
                )
                stoch = post["stoch"]
                stochs.append(stoch)
                deters.append(deter)
                post_mean.append(post["mean"])
                post_std.append(post["std"])
                prior_mean.append(prior["mean"])
                prior_std.append(prior["std"])
            stochs = tf.stack(stochs, axis=0)
            deters = tf.stack(deters, axis=0)
            post_mean = tf.stack(post_mean, axis=0)
            post_std = tf.stack(post_std, axis=0)
            prior_mean = tf.stack(prior_mean, axis=0)
            prior_std = tf.stack(prior_std, axis=0)

            # (seq, batch, shape) -> (batch, seq, shape)
            stochs = tf.transpose(stochs, [1, 0, 2])
            deters = tf.transpose(deters, [1, 0, 2])
            post_mean = tf.transpose(post_mean, [1, 0, 2])
            post_std = tf.transpose(post_std, [1, 0, 2])
            prior_mean = tf.transpose(prior_mean, [1, 0, 2])
            prior_std = tf.transpose(prior_std, [1, 0, 2])

            feat = tf.concat([stochs, deters], -1)
            feat = tf.reshape(feat, (self.config.batch_size * self.config.batch_length,) + feat.shape[2:])
            image_pred = self.parameter.decode(feat)
            reward_pred = self.parameter.reward(feat)

            image_loss = tf.reduce_mean(image_pred.log_prob(states))
            reward_loss = tf.reduce_mean(reward_pred.log_prob(rewards))

            prior_dist = tfd.MultivariateNormalDiag(prior_mean, prior_std)
            post_dist = tfd.MultivariateNormalDiag(post_mean, post_std)

            kl_loss = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            kl_loss = tf.maximum(kl_loss, self.config.free_nats)
            loss = self.config.kl_scale * kl_loss - image_loss - reward_loss

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

            info["img_loss"] = -image_loss.numpy() / (64 * 64 * 3)
            info["reward_loss"] = -reward_loss.numpy()
            info["kl_loss"] = kl_loss.numpy()

        if (not self.config.enable_train_actor) and (not self.config.enable_train_value):
            # WorldModelsのみ学習
            self.train_count += 1
            return info

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
        horizon_v = None
        if self.config.enable_train_actor:
            self.parameter.actor.trainable = True
            self.parameter.value.trainable = False
            with tf.GradientTape() as tape:
                horizon_v, horizon_feats = self._compute_horizon_step(stochs, deters, feats)

                # (horizon, batch_size*batch_length, 1) -> (batch_size*batch_length, horizon, 1)
                act_loss = tf.transpose(horizon_v, [1, 0, 2])
                act_loss = tf.reduce_sum(act_loss, axis=1) / self.config.horizon
                act_loss = -tf.reduce_mean(act_loss)

            grads = tape.gradient(act_loss, self.parameter.actor.trainable_variables)
            self._actor_opt.apply_gradients(zip(grads, self.parameter.actor.trainable_variables))
            info["act_loss"] = act_loss.numpy()

        # ------------------------
        # Value
        # ------------------------
        if self.config.enable_train_value:
            if horizon_v is None:
                horizon_v, horizon_feats = self._compute_horizon_step(stochs, deters, feats)

            # (horizon, batch_size*batch_length, feat) -> (horizon*batch_size*batch_length, feat)
            horizon_feats = tf.stack(horizon_feats)
            horizon_feats = tf.reshape(
                horizon_feats, (horizon_feats.shape[0] * horizon_feats.shape[1], horizon_feats.shape[2])
            )
            horizon_v = tf.reshape(horizon_v, (horizon_v.shape[0] * horizon_v.shape[1], horizon_v.shape[2]))

            self.parameter.actor.trainable = False
            self.parameter.value.trainable = True
            with tf.GradientTape() as tape:
                value_pred = self.parameter.value(horizon_feats)
                val_loss = -tf.reduce_mean(value_pred.log_prob(horizon_v))
            grads = tape.gradient(val_loss, self.parameter.value.trainable_variables)
            self._value_opt.apply_gradients(zip(grads, self.parameter.value.trainable_variables))
            info["val_loss"] = val_loss.numpy()

        self.train_count += 1
        return info

    def _compute_horizon_step(
        self,
        stochs,
        deters,
        feats,
    ):
        if self.config.value_estimation_method == "simple":
            horizon_feats = []
            horizon_reward = []
            for t in range(self.config.horizon):
                stochs, deters, feats = self._horizon_step(stochs, deters, feats)
                horizon_feats.append(feats)
                horizon_reward.append(self.parameter.reward(feats).mode())

            # 累積和の平均
            horizon_v = tf.math.cumsum(horizon_reward, reverse=True)
            weights = tf.reshape(1.0 / tf.range(len(horizon_v), 0, -1, dtype=tf.float32), (len(horizon_v), 1, 1))
            weights = tf.tile(weights, (1, horizon_v.shape[1], horizon_v.shape[2]))
            horizon_v *= weights

            return horizon_v, horizon_feats

        elif self.config.value_estimation_method == "dreamer":
            horizon_feats = []
            horizon_v = []
            for t in range(self.config.horizon):
                stochs, deters, feats = self._horizon_step(stochs, deters, feats)
                horizon_feats.append(feats)
                reward = self.parameter.reward(feats).mode()
                v = self.parameter.value(feats).mode()
                horizon_v.append(
                    (self.config.discount**t) * reward + (self.config.discount ** (self.config.horizon - t)) * v
                )

            horizon_v = tf.math.cumsum(horizon_v, reverse=True)
            weights = tf.reshape(1.0 / tf.range(len(horizon_v), 0, -1, dtype=tf.float32), (len(horizon_v), 1, 1))
            weights = tf.tile(weights, (1, horizon_v.shape[1], horizon_v.shape[2]))
            horizon_v *= weights

            # EWA
            v = (1 - self.config.disclam) * horizon_v[0]
            horizon_v2 = [v]
            for t in range(1, self.config.horizon):
                v = self.config.disclam * v + (1 - self.config.disclam) * horizon_v[t]
                horizon_v2.append(v)

            horizon_v2 = tf.stack(horizon_v2)
            return horizon_v2, horizon_feats

        else:
            raise ValueError(self.config.value_estimation_method)

    def _horizon_step(
        self,
        stoch,
        deter,
        feat,
    ):
        # --- policy
        policy = self.parameter.actor(feat).sample()

        # --- step
        deter, prior = self.parameter.dynamics.img_step(stoch, deter, policy)
        stoch = prior["stoch"]
        feat = tf.concat([stoch, deter], -1)

        return stoch, deter, feat


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

        self._recent_states = []
        self._recent_actions = []
        self._recent_rewards = []

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        if self.config.experience_acquisition_method != "loop":
            self._recent_states = []
            self._recent_actions = []
            self._recent_rewards = []

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
            epsilon = self.config.epsilon
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
        latent, deter, _ = self.parameter.dynamics.obs_step(self.stoch, self.deter, prev_action, embed)
        self.feat = tf.concat([latent["stoch"], deter], axis=1)
        self.deter = deter
        self.stoch = latent["stoch"]

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
            self._recent_states.append(next_state)
            self._recent_actions.append(self.action)
            self._recent_rewards.append(reward)
            if len(self._recent_states) == self.config.batch_length:
                self.remote_memory.add(
                    {
                        "states": self._recent_states,
                        "actions": self._recent_actions,
                        "rewards": self._recent_rewards,
                    }
                )
                self._recent_states = []
                self._recent_actions = []
                self._recent_rewards = []
        else:
            if len(self._recent_states) < self.config.batch_length:
                self._recent_states.append(next_state)
                self._recent_actions.append(self.action)
                self._recent_rewards.append(reward)

            if done:
                for _ in range(self.config.batch_length - len(self._recent_states)):
                    self._recent_states.append(next_state)
                    self._recent_actions.append(random.randint(0, self.config.action_num - 1))
                    self._recent_rewards.append(reward)

                self.remote_memory.add(
                    {
                        "states": self._recent_states,
                        "actions": self._recent_actions,
                        "rewards": self._recent_rewards,
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
        HEIGHT = (IMG_H + PADDING + STR_H * 2) * (_view_sample + 1) + 5

        if self.screen is None:
            self.screen = pw.create_surface(WIDTH, HEIGHT)
        pw.draw_fill(self.screen, color=(0, 0, 0))

        if self.feat is None:
            self._rssm_step()

        # --- decode
        pred_state = self.parameter.decode(self.feat).mode()[0].numpy()  # type:ignore , ignore check "None"
        rmse = np.sqrt(np.mean((self.state - pred_state) ** 2))

        pred_reward = self.parameter.reward(self.feat).mode()[0][0].numpy()  # type:ignore , ignore check "None"
        pred_value = self.parameter.value(self.feat).mode()[0][0].numpy()
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
            value_dist = self.parameter.value(feat)

            # 縦にいくつかサンプルを表示
            for j in range(_view_sample):
                if j == 0:
                    next_state = next_state_dist.mode()  # type:ignore , ignore check "None"
                    reward = reward_dist.mode()  # type:ignore , ignore check "None"
                    value = value_dist.mode()  # type:ignore , ignore check "None"
                else:
                    next_state = next_state_dist.sample()  # type:ignore , ignore check "None"
                    reward = reward_dist.sample()  # type:ignore , ignore check "None"
                    value = value_dist.sample()  # type:ignore , ignore check "None"

                n_img = next_state[0].numpy() * 255
                reward = reward.numpy()[0][0]
                value = value.numpy()[0][0]

                x = (IMG_W + PADDING) * a
                y = 20 + IMG_H + STR_H + (IMG_H + PADDING + STR_H * 2) * j
                pw.draw_text(self.screen, x, y, f"r={reward:.3f}", color=(255, 255, 255))
                pw.draw_text(self.screen, x, y + STR_H, f"V={value:.3f}", color=(255, 255, 255))
                pw.draw_image_rgb_array(self.screen, x, y + STR_H * 2, n_img)

        return pw.get_rgb_array(self.screen)
