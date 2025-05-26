from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.rl import functions as funcs
from srl.rl.tf.distributions.categorical_gumbel_dist_block import CategoricalGumbelDistBlock
from srl.rl.tf.distributions.normal_dist_block import NormalDistBlock
from srl.rl.tf.model import KerasModelAddedSummary
from srl.utils.common import compare_less_version

from .config import Config

kl = keras.layers
v216_older = compare_less_version(tf.__version__, "2.16.0")

Conv2D3x3 = partial(kl.Conv2D, kernel_size=3, strides=1, padding="same", use_bias=False)
IdentityLayer = partial(kl.Lambda, function=lambda x: x)  # 何もしないレイヤー


class ResidualBlock(KerasModelAddedSummary):
    def __init__(
        self,
        filters: int,
        downsample: bool = False,
        momentum: float = 0.9,  # torchとtfで逆 torch:0.1 == tf:0.9
        **kwargs,
    ):
        super().__init__(**kwargs)

        if downsample:
            strides = 2
            self.downsample = kl.Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                strides=2,
                padding="same",
                use_bias=False,
            )
        else:
            strides = 1
            self.downsample = IdentityLayer()

        self.h_layers = [
            Conv2D3x3(filters, strides=strides),
            kl.BatchNormalization(momentum=momentum),
            kl.Activation("relu"),
            Conv2D3x3(filters),
            kl.BatchNormalization(momentum=momentum),
        ]
        self.out_act = kl.Activation("relu")

    def call(self, x, training=False):
        identity = x
        identity = self.downsample(identity)

        for h in self.h_layers:
            x = h(x, training=training)

        x = x + identity
        x = self.out_act(x)
        return x


class DownSample(KerasModelAddedSummary):
    def __init__(
        self,
        out_channels,
        momentum,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.h_layers = [
            kl.Conv2D(out_channels // 2, kernel_size=3, strides=1, padding="same", use_bias=False),
            kl.BatchNormalization(momentum=momentum),
            kl.Activation("relu"),
            ResidualBlock(out_channels // 2, momentum=momentum),
            ResidualBlock(out_channels, downsample=True, momentum=momentum),
            ResidualBlock(out_channels, momentum=momentum),
            kl.AveragePooling2D(pool_size=3, strides=2, padding="same"),
            ResidualBlock(out_channels, momentum=momentum),
            kl.AveragePooling2D(pool_size=3, strides=2, padding="same"),
        ]

    def call(self, x, training=False):
        for h in self.h_layers:
            x = h(x, training=training)
        return x


class RepresentationNetwork(KerasModelAddedSummary):
    def __init__(self, in_shape, cfg: Config, **kwargs):
        super().__init__(**kwargs)
        num_blocks = cfg.res_blocks
        num_channels = cfg.res_channels
        downsample = cfg.downsample
        momentum = cfg.normalize_momentam
        np_dtype = cfg.get_dtype("np")

        if downsample:
            self.h_layers = [
                DownSample(num_channels, momentum=momentum),
            ]
        else:
            self.h_layers = [
                Conv2D3x3(num_channels),
                kl.BatchNormalization(momentum=momentum),
                kl.Activation("relu"),
            ]
        self.h_layers += [ResidualBlock(num_channels, momentum=momentum) for _ in range(num_blocks)]

        # build & 出力shapeを取得
        s_state = self(np.zeros(shape=(1,) + in_shape, dtype=np_dtype))
        self.s_state_shape = s_state.shape[1:]

    def call(self, x, training=False):
        for h in self.h_layers:
            x = h(x, training=training)
        return x


class DynamicsNetwork(KerasModelAddedSummary):
    def __init__(self, s_state_shape, cfg: Config, **kwargs):
        super().__init__(**kwargs)
        np_dtype = cfg.get_dtype("np")
        self.tf_dtype = cfg.get_dtype("tf")
        num_blocks = cfg.res_blocks
        num_channels = cfg.res_channels
        reward_units = cfg.reward_units
        self.reward_range_num = cfg.reward_range_num
        self.reward_range = cfg.reward_range
        momentum = cfg.normalize_momentam
        np_dtype = cfg.get_dtype("np")

        # --- action
        if isinstance(cfg.action_space, DiscreteSpace):
            self.act_emb = kl.Embedding(cfg.action_space.n, num_channels)
            action_shape = ()
        else:
            self.act_emb = kl.Dense(num_channels, activation="relu")
            action_shape = (cfg.action_space.size,)

        # --- state block
        self.img_layers = [
            Conv2D3x3(num_channels),
            kl.BatchNormalization(momentum=momentum),
        ]
        self.img_act = kl.Activation("relu")
        self.img_resblocks = [ResidualBlock(num_channels, momentum=momentum) for _ in range(num_blocks)]

        # --- reward block
        self.reward_layers = [
            kl.Conv2D(reward_units, kernel_size=(1, 1), padding="same"),
            kl.BatchNormalization(momentum=momentum),
            kl.ReLU(),
            kl.Flatten(),
        ]
        self.lstm = kl.LSTM(reward_units, return_sequences=True, return_state=True)
        self.value_prefix_layers = [
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Dense(reward_units),
            kl.BatchNormalization(momentum=momentum),
            kl.ReLU(),
            kl.Dense(
                self.reward_range_num,
                activation="softmax",
                kernel_initializer="zeros",
                bias_initializer="zeros",
            ),
        ]

        # build
        self(
            [
                np.zeros((1,) + s_state_shape, dtype=np_dtype),
                np.zeros((1,) + action_shape),
                self.get_initial_state(),
            ]
        )

    def get_initial_state(self, batch_size: int = 1):
        if v216_older:
            hc = self.lstm.get_initial_state(None, batch_size, dtype=self.tf_dtype)
        else:
            hc = self.lstm.get_initial_state(batch_size)
        return hc

    def call(self, inputs, training=False):
        x = inputs[0]
        action = tf.cast(inputs[1], self.tf_dtype)
        hc = inputs[2]

        # --- action
        b, h, w, ch = x.shape
        if self.act_emb is not None:
            # (b) -> (b, ch)
            action = self.act_emb(action, training=training)
        action = tf.broadcast_to(
            tf.reshape(action, (b, 1, 1, -1)),  # (b, 1, 1, ch)
            (b, h, w, action.shape[-1]),  # (b, h, w, ch)
        )

        # --- state block
        x_skip = x  # skipはaction部分は除く
        x = tf.concat([x, action], axis=-1)
        for h in self.img_layers:
            x = h(x, training=training)
        x = x + x_skip
        x = self.img_act(x)
        for h in self.img_resblocks:
            x = h(x, training=training)
        state = x

        # --- reward block
        for h in self.reward_layers:
            x = h(x, training=training)

        # lstm
        x = x[:, tf.newaxis, ...]
        value_prefix, hx, cx = self.lstm(x, initial_state=hc, training=training)
        value_prefix = value_prefix[:, 0, ...]

        for h in self.value_prefix_layers:
            value_prefix = h(value_prefix, training=training)

        return state, value_prefix, (hx, cx)

    def pred(self, s_state, action, hc):
        n_state, value_prefix, hc = self([s_state, action, hc])
        value_prefix = funcs.twohot_decode(
            value_prefix.numpy(),
            self.reward_range_num,
            self.reward_range[0],
            self.reward_range[1],
        )
        return n_state, value_prefix, hc


class PredictionNetwork(KerasModelAddedSummary):
    def __init__(self, s_state_shape, cfg: Config, **kwargs):
        super().__init__(**kwargs)
        policy_units = cfg.policy_units
        value_units = cfg.value_units
        self.value_range_num = cfg.value_range_num
        self.value_range = cfg.value_range
        momentum = cfg.normalize_momentam
        np_dtype = cfg.get_dtype("np")

        # --- policy
        self.policy_layers = [
            kl.Conv2D(1, kernel_size=(1, 1), padding="same"),
            kl.BatchNormalization(momentum=momentum),
            kl.ReLU(),
            kl.Flatten(),
            kl.Dense(policy_units),
            kl.BatchNormalization(momentum=momentum),
            kl.ReLU(),
        ]
        if isinstance(cfg.action_space, DiscreteSpace):
            self.policy_layers.append(CategoricalGumbelDistBlock(cfg.action_space.n))
        elif isinstance(cfg.action_space, ArrayContinuousSpace):
            self.policy_layers.append(NormalDistBlock(cfg.action_space.size, enable_squashed=True))

        # --- value
        self.value_layers = [
            kl.Conv2D(1, kernel_size=(1, 1), padding="same"),
            kl.BatchNormalization(momentum=momentum),
            kl.ReLU(),
            kl.Flatten(),
            kl.Dense(value_units),
            kl.BatchNormalization(momentum=momentum),
            kl.ReLU(),
            kl.Dense(
                self.value_range_num,
                activation="softmax",
                kernel_initializer="zeros",
                bias_initializer="zeros",
            ),
        ]

        # build
        self(np.zeros((1,) + s_state_shape, dtype=np_dtype))

    def call(self, state, training=False):
        policy = state
        for layer in self.policy_layers:
            policy = layer(policy, training=training)

        value = state
        for layer in self.value_layers:
            value = layer(value, training=training)

        return policy, value

    def pred(self, s_state):
        policy, value = self(s_state)
        value = funcs.twohot_decode(
            value.numpy(),
            self.value_range_num,
            self.value_range[0],
            self.value_range[1],
        )
        return policy, value


class ProjectorNetwork(KerasModelAddedSummary):
    def __init__(self, in_shape, cfg: Config):
        super().__init__()
        np_dtype = cfg.get_dtype("np")

        # projection
        self.projection_layers = [
            kl.Dense(cfg.projection_hid),
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Dense(cfg.projection_hid),
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Dense(cfg.projection_out),
            kl.BatchNormalization(),
        ]
        self.projection_head_layers = [
            kl.Dense(cfg.projection_head_hid),
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Dense(cfg.projection_head_out),
        ]

        # --- build
        self(np.zeros(shape=(1,) + in_shape, dtype=np_dtype))

    def call(self, x, training=False):
        for h in self.projection_layers:
            x = h(x, training=training)
        for h in self.projection_head_layers:
            x = h(x, training=training)
        return x

    def projection(self, x, training=False):
        for h in self.projection_layers:
            x = h(x, training=training)
        return x
