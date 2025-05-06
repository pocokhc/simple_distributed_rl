import math

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.rl.tf.model import KerasModelAddedSummary

from .config import DenoiserConfig
from .model_unet import Conv2D3x3, GroupNorm, UNet

kl = keras.layers


class FourierFeatures(keras.layers.Layer):
    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(1, self.dim // 2), initializer=keras.initializers.RandomNormal(), trainable=False, name="weight")
        super().build(input_shape)

    def call(self, inputs):
        inputs = tf.reshape(inputs, [-1])  # Flatten the inputs to handle any shape
        f = 2.0 * math.pi * tf.expand_dims(inputs, axis=-1) @ self.weight
        return tf.concat([tf.cos(f), tf.sin(f)], axis=-1)


class Denoiser(KerasModelAddedSummary):
    def __init__(self, img_shape, action_num: int, cfg: DenoiserConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.num_steps_conditioning = cfg.num_steps_conditioning
        assert cfg.condition_channels % 2 == 0
        c_ch = cfg.condition_channels

        # --- condition
        self.noise_emb = FourierFeatures(c_ch)
        self.act_emb = kl.Embedding(action_num, c_ch)
        self.act_flatten = kl.Flatten()
        self.cond_concat = kl.Concatenate(-1)
        self.cond_dense1 = kl.Dense(c_ch, activation="silu")
        self.cond_dense2 = kl.Dense(c_ch)
        self.cond_obs_concat = kl.Concatenate(-1)

        # --- denoiser
        self.conv_in = Conv2D3x3(cfg.channels_list[0])
        self.unet = UNet(cfg.channels_list, cfg.res_block_num_list, cfg.use_attention_list)
        self.out_norm = GroupNorm()
        self.out_act = kl.Activation("silu")
        self.out_conv = Conv2D3x3(img_shape[-1], kernel_initializer="zeros")

        self.loss_fn = keras.losses.Huber(reduction="none")
        self.optimizer = keras.optimizers.AdamW(learning_rate=cfg.lr)

        # --- build
        recent_obs = np.zeros((1, self.num_steps_conditioning, img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
        recent_act = np.zeros((1, self.num_steps_conditioning), dtype=np.int64)
        self(
            [
                np.zeros((1,) + img_shape, dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
                recent_obs,
                recent_act,
            ]
        )

    def call(self, x, training=False):
        noisy_obs = x[0]
        c_noise = x[1]
        recent_obs = x[2]
        recent_act = x[3]

        act_emb = self.act_emb(recent_act, training=training)  # (b,t) -> (b,t,e)
        act_emb = self.act_flatten(act_emb)  # b t e -> b (t*e)

        # condition
        noise_emb = self.noise_emb(c_noise, training=training)

        condition = self.cond_concat([noise_emb, act_emb])
        condition = self.cond_dense1(condition)
        condition = self.cond_dense2(condition)

        # recent_obs, (b, t, h, w, ch) -> (b, h, w, ch*t)
        b, t, h, w, c = recent_obs.shape
        recent_obs = tf.reshape(tf.transpose(recent_obs, perm=[0, 2, 3, 4, 1]), (b, h, w, c * t))

        # recent_obsはchとして追加
        x = self.cond_obs_concat([noisy_obs, recent_obs])
        x = self.conv_in(x)
        x, _, _ = self.unet([x, condition])
        x = self.out_norm(x, training=training)
        x = self.out_act(x, training=training)
        x = self.out_conv(x, training=training)
        return x

    def update(self, obs, recent_obs, recent_act):
        # obs.shape: (batch, h, w, ch)
        batch_size, _, _, _ = obs.shape

        # sample Noise distibution
        sigma = np.exp(self.cfg.noise_mean + self.cfg.noise_std * tf.random.normal(shape=(batch_size, 1, 1, 1)))

        # add noise
        offset_noise = self.cfg.sigma_offset_noise * tf.random.normal(shape=obs.shape)
        noise = tf.random.normal(shape=obs.shape)
        noisy_obs = obs + offset_noise + noise * sigma

        weight = (sigma**2 + self.cfg.sigma_data**2) / (sigma * self.cfg.sigma_data) ** 2
        with tf.GradientTape() as tape:
            denoised_obs = self.denoise(noisy_obs, sigma, recent_obs, recent_act, training=True)
            loss = weight * self.loss_fn(obs, denoised_obs)[..., tf.newaxis]
            loss = tf.reduce_mean(loss)
            loss += tf.reduce_sum(self.losses)  # layer正則化項
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return loss.numpy()

    @tf.function
    def denoise(self, noisy_img, sigma, recent_obs, recent_act, training=False):
        sigma = tf.sqrt(sigma**2 + self.cfg.sigma_offset_noise**2)

        # 正規化
        c_in = 1 / tf.sqrt(sigma**2 + self.cfg.sigma_data**2)
        scaled_noisy_img = c_in * noisy_img
        scaled_recent_obs = recent_obs / self.cfg.sigma_data
        c_noise = tf.math.log(sigma) / 4

        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * tf.sqrt(c_skip)
        network_output = self([scaled_noisy_img, c_noise, scaled_recent_obs, recent_act], training=training)
        return c_skip * noisy_img + c_out * network_output
