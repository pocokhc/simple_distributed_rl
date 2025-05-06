import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.rl.tf.distributions.categorical_dist_block import CategoricalDist
from srl.rl.tf.model import KerasModelAddedSummary
from srl.utils.common import compare_less_version

from .config import ActorCriticConfig
from .model_unet import Conv2D3x3, SmallResBlock

kl = keras.layers
v216_older = compare_less_version(tf.__version__, "2.16.0")


class ActorCritic(KerasModelAddedSummary):
    def __init__(self, img_shape, action_num: int, cfg: ActorCriticConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.action_num = action_num

        if not (len(cfg.channels_list) == len(cfg.enable_downsampling_list)):
            raise ValueError(f"{len(cfg.channels_list)} == {len(cfg.enable_downsampling_list)}")

        # --- build
        self(np.zeros((1,) + img_shape, dtype=np.float32))

    def get_initial_state(self, batch_size=1):
        if v216_older:
            return self.lstm.cell.get_initial_state(batch_size=batch_size, dtype=self.dtype)
        else:
            return self.lstm.cell.get_initial_state(batch_size)

    def build(self, input_shape):
        cfg = self.cfg

        # --- cnn encoder
        self.encoder_layers = [Conv2D3x3(cfg.channels_list[0])]
        for i in range(len(cfg.channels_list)):
            self.encoder_layers.append(SmallResBlock(cfg.channels_list[i]))
            if cfg.enable_downsampling_list[i]:
                self.encoder_layers.append(kl.MaxPool2D(2))

        # lstm
        self.lstm = kl.LSTM(cfg.lstm_dim, return_sequences=True, return_state=True)

        # output layer
        self.critic_linear = kl.Dense(
            1,
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )
        self.actor_linear = kl.Dense(
            self.action_num,
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )

        self.built = True

    def call(self, x, hc=None, training=False):
        for h in self.encoder_layers:
            x = h(x, training=training)

        # (b, h, w, c) -> (b, 1, h*w*c)
        x = tf.reshape(x, (tf.shape(x)[0], 1, -1))

        x, hx, cx = self.lstm(x, initial_state=hc, training=training)
        x = tf.squeeze(x, axis=1)

        act = self.actor_linear(x, training=training)
        v = self.critic_linear(x, training=training)
        return CategoricalDist(act), v, (hx, cx)
