from typing import List

import tensorflow as tf
from tensorflow import keras

from .config import DiffusionSamplerConfig
from .model_denoiser import Denoiser

kl = keras.layers


class DiffusionSampler:
    def __init__(self, denoiser: Denoiser, cfg: DiffusionSamplerConfig, tf_dtype):
        self.denoiser = denoiser
        self.cfg = cfg
        self.tf_dtype = tf_dtype
        # 生成回数は固定なのでスケジュールを事前に作成
        self.sigmas = self.create_timesteps()

    def create_timesteps(self) -> List[float]:
        if self.cfg.num_steps_denoising == 1:
            return [self.cfg.sigma_max, 0]
        min_inv_rho = self.cfg.sigma_min ** (1 / self.cfg.rho)
        max_inv_rho = self.cfg.sigma_max ** (1 / self.cfg.rho)
        N = self.cfg.num_steps_denoising
        t = [
            (max_inv_rho + i / (N - 1) * (min_inv_rho - max_inv_rho)) ** self.cfg.rho
            for i in range(N)  #
        ]
        t += [0]
        return t

    def sample(self, recent_obs, recent_act):
        b, t, h, w, ch = recent_obs.shape

        gamma_ = min(self.cfg.s_churn / self.cfg.num_steps_denoising, 2**0.5 - 1)

        x = tf.random.normal((b, h, w, ch)) * self.sigmas[0]
        trajectory = [x]
        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.cfg.s_min <= sigma <= self.cfg.s_max else 0
            sigma_hat = sigma * (gamma + 1)

            eps = tf.random.normal((b, h, w, ch)) * self.cfg.s_noise
            x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5

            tf_sigma_hat = tf.convert_to_tensor([sigma_hat] * b, dtype=self.tf_dtype)[..., tf.newaxis, tf.newaxis, tf.newaxis]
            denoised_img = self.denoiser.denoise(x, tf_sigma_hat, recent_obs, recent_act)

            d = (x - denoised_img) / sigma_hat
            dt = next_sigma - sigma_hat

            if self.cfg.order == 1 or next_sigma == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                tf_next_sigma = tf.convert_to_tensor([next_sigma] * b, dtype=self.tf_dtype)[..., tf.newaxis, tf.newaxis, tf.newaxis]
                denoised_img_2 = self.denoiser.denoise(x_2, tf_next_sigma, recent_obs, recent_act)
                d_dash = (x_2 - denoised_img_2) / next_sigma
                x = x + dt * (d + d_dash) / 2

            trajectory.append(x)
        return x, trajectory
