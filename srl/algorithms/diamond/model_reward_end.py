import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.rl.tf.model import KerasModelAddedSummary
from srl.utils.common import compare_less_version

from .config import RewardEndModelConfig
from .model_unet import Conv2D3x3, Downsample, IdentityLayer, ResBlocks

kl = keras.layers
v216_older = compare_less_version(tf.__version__, "2.16.0")


class RewardEndModel(KerasModelAddedSummary):
    def __init__(self, img_shape, action_num: int, cfg: RewardEndModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.action_num = action_num

        if not (len(cfg.channels_list) == len(cfg.res_block_num_list) and len(cfg.res_block_num_list) == len(cfg.use_attention_list)):
            raise ValueError(f"{len(cfg.channels_list)} == {len(cfg.res_block_num_list)} == {len(cfg.use_attention_list)}")

        self.reward_loss = keras.losses.CategoricalCrossentropy(from_logits=True)
        self.done_loss = keras.losses.CategoricalCrossentropy(from_logits=True)

        # --- build
        self(
            [
                np.zeros((1, 1) + img_shape, dtype=np.float32),
                np.zeros((1, 1), dtype=np.int64),
                np.zeros((1, 1) + img_shape, dtype=np.float32),
            ]
        )

    def build(self, input_shape):
        # b, t, h, w, c = input_shape[0]
        cfg = self.cfg

        # --- action embedding
        self.act_emb = kl.Embedding(self.action_num, cfg.condition_channels)

        # --- cnn encoder
        self.conv_concat = kl.Concatenate(axis=-1)
        self.conv_in = Conv2D3x3(cfg.channels_list[0])

        self.downsamples = []
        self.blocks = []
        for i in range(len(cfg.channels_list)):
            self.downsamples.append(Downsample() if i != 0 else IdentityLayer())
            self.blocks.append(
                ResBlocks(
                    [cfg.channels_list[i]] * cfg.res_block_num_list[i],
                    use_attention=cfg.use_attention_list[i],
                ),
            )
        self.downsamples.append(IdentityLayer())
        self.blocks.append(
            ResBlocks(
                channels_list=[cfg.channels_list[-1]] * 2,
                use_attention=True,
            )
        )

        # --- lstm
        self.lstm = kl.LSTM(cfg.lstm_dim, return_sequences=True, return_state=True)

        # --- reward, done
        self.mid_layer = kl.Dense(cfg.lstm_dim, activation="silu")
        self.reward_layer = kl.Dense(3, use_bias=False)
        self.done_layer = kl.Dense(2, use_bias=False)

        self.built = True

    def get_initial_state(self, batch_size=1):
        if v216_older:
            return self.lstm.cell.get_initial_state(batch_size=batch_size, dtype=self.dtype)
        else:
            return self.lstm.cell.get_initial_state(batch_size)

    def call(self, x, hc=None, training=False):
        obs = x[0]
        act = x[1]
        next_obs = x[2]
        b, t, h, w, c = obs.shape

        # act emb
        # (b,t)  -> (b*t) -> (b*t, act_emb)
        act = tf.reshape(act, (b * t,))
        condition = self.act_emb(act)

        # cnn encoder
        # (b, t, h, w, c) -> (b*t, h, w, c)
        obs = tf.reshape(obs, (b * t, h, w, c))
        next_obs = tf.reshape(next_obs, (b * t, h, w, c))
        x = self.conv_concat([obs, next_obs])
        x = self.conv_in(x)
        for block, down in zip(self.blocks, self.downsamples):
            x = down(x)
            x, _ = block([x, condition])
        # (b*t, h, w, c) -> (b, t, h*w*c)
        x = tf.reshape(x, (b, t, -1))

        x, hx, cx = self.lstm(x, initial_state=hc, training=training)

        x = self.mid_layer(x, training=training)
        r = self.reward_layer(x, training=training)
        d = self.done_layer(x, training=training)
        return r, d, (hx, cx)

    @tf.function
    def compute_train_loss(self, obs, act, next_obs, reward, done, hc):
        r, d, hc = self([obs, act, next_obs], hc, training=True)
        loss_r = self.reward_loss(reward, r)
        loss_d = self.done_loss(done, d)
        return loss_r, loss_d, hc
