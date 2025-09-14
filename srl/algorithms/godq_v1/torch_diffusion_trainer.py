import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.rl.trainer import RLTrainer

from .config import DenoiserConfig
from .memory import Memory
from .torch_diffusion import Denoiser


class DiffusionTrainer:
    def on_setup(self, trainer: RLTrainer):
        self.info = trainer.info
        self.train_count = 0
        self.cfg: DenoiserConfig = trainer.config.denoiser

        self.memory: Memory = trainer.memory
        self.denoiser: Denoiser = trainer.parameter.denoiser
        self.device = torch.device(trainer.config.used_device_torch)
        self.np_dtype = trainer.config.get_dtype("np")

        self.loss_fn = nn.HuberLoss(reduction="none")
        self.optimizer = optim.AdamW(self.denoiser.parameters(), lr=self.cfg.lr)

        self.batch_size = trainer.config.batch_size
        img_shape = trainer.config.obs_render_img_space.shape
        self.states_np = np.empty((self.batch_size, *img_shape), dtype=self.np_dtype)
        self.n_states_np = np.empty((self.batch_size, *img_shape), dtype=self.np_dtype)
        self.action_indices_np = np.empty((self.batch_size,), dtype=np.int64)

    def train(self):
        batches = self.memory.sample_diff()
        if batches is None:
            return

        for i, b in enumerate(batches):
            self.states_np[i] = b[0]
            self.n_states_np[i] = b[1]
            self.action_indices_np[i] = b[2]
        state = torch.from_numpy(self.states_np).to(self.device)
        n_state = torch.from_numpy(self.n_states_np).to(self.device)
        action_index = torch.from_numpy(self.action_indices_np).to(self.device)

        # --- sample Noise distibution
        sigma = torch.exp(self.cfg.noise_mean + self.cfg.noise_std * torch.randn(self.batch_size, 1, 1, 1, device=self.device))

        # --- add noise
        # offset_noise = self.cfg.sigma_offset_noise * torch.randn_like(obs)
        # noise = torch.randn_like(obs)
        # noisy_obs = obs + offset_noise + noise * sigma
        noisy_obs = n_state + torch.randn_like(n_state) * sigma
        weight = (sigma**2 + self.cfg.sigma_data**2) / (sigma * self.cfg.sigma_data) ** 2

        # loss
        denoised_obs = self.denoiser.denoise(noisy_obs, sigma, state, action_index)
        loss = (weight * self.loss_fn(n_state, denoised_obs)).mean()

        # --- bp
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), max_norm=self.cfg.max_grad_norm)
        self.optimizer.step()
        self.info["loss_denoiser"] = loss.item()

        self.train_count += 1
