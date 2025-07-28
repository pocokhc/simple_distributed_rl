import logging
from itertools import chain
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.rl.trainer import RLTrainer
from srl.rl.torch_.helper import model_params_soft_sync, model_soft_sync

from .config import Config
from .memory import Memory
from .torch_model import Model

logger = logging.getLogger(__name__)


def reset_model_params(
    params: Iterable[nn.Parameter],
    shrink_rate: float,
    perturb_rate: float,
    stddev: float,
):
    for param in params:
        base = shrink_rate * param.data
        delta = perturb_rate * param.data
        noise = torch.tanh(torch.randn_like(param) * stddev) * delta
        param.data.copy_(base + noise)


def shrink_model_params(params: Iterable[nn.Parameter], rate: float):
    for param in params:
        param.data.copy_(rate * param.data)


class TorchTrainer:
    def on_setup(self, trainer: RLTrainer):
        self.trainer = trainer
        self.config: Config = trainer.config
        self.memory: Memory = trainer.memory
        self.net: Model = trainer.parameter.net
        self.torch_dtype = self.config.get_dtype("torch")
        self.np_dtype = self.config.get_dtype("np")

        # --- reset params
        self.shrink_params = []

        # --- q
        self.models = [self.net.q_online]
        self.loss_q_func = nn.HuberLoss(reduction="none")
        enc_params = list(self.net.q_online.encoder.parameters())
        self.shrink_params += enc_params[1:]

        # --- rnd
        if self.config.discount <= 0:
            self.net.rnd_train.train()
            self.opt_rnd = optim.Adam(self.net.rnd_train.parameters(), lr=self.config.lr)

        if self.config.feat_type == "SimSiam":
            self.models.append(self.net.projector)
            self.shrink_params += list(self.net.projector.parameters())
            self.loss_feat_func = nn.HuberLoss()
        elif self.config.feat_type == "SPR":
            self.models.append(self.net.spr)
            self.shrink_params += list(self.net.spr.parameters())
            self.loss_feat_func = nn.HuberLoss()

        if self.config.enable_archive:
            self.models.append(self.net.latent_encoder)
            self.shrink_params += list(self.net.latent_encoder.parameters())
            self.loss_lat_func = nn.HuberLoss()

        self.params = list(chain(*[m.parameters() for m in self.models]))
        if self.config.replay_ratio > 1:
            self.opt = optim.AdamW(self.params, lr=self.config.lr, weight_decay=0.1)
        else:
            self.opt = optim.RAdam(self.params, lr=self.config.lr)
        [m.train() for m in self.models]

        self.reset_shrink = 0

    def train(self):
        for _ in range(self.config.replay_ratio):
            batches = self.memory.sample_q()
            if batches is None:
                break

            # --- reset
            if (self.config.reset_interval_shrink > 0) and (self.trainer.train_count % self.config.reset_interval_shrink == 1):
                shrink_model_params(self.shrink_params, (1 - self.config.lr))
                self.reset_shrink += 1
                self.trainer.info["reset_shrink"] = self.reset_shrink

            # --- train
            self._train(batches)

            # --- targetと同期
            model_params_soft_sync(self.net.q_target.parameters(), self.net.q_online.parameters(), self.config.target_model_update_rate)
            if self.config.feat_type == "SPR":
                model_soft_sync(self.net.project_target_block, self.net.spr.project_block, self.config.target_model_update_rate)

            self.trainer.train_count += 1

    def _train(self, batches):
        device = self.net.device
        batches, weights, update_args = batches
        if self.config.memory.requires_priority():
            weights = torch.tensor(np.asarray(weights), dtype=self.torch_dtype, device=device)

        state, action, n_state, reward, not_terminated = zip(*batches)
        state = torch.tensor(np.asarray(state, dtype=self.np_dtype), device=device)
        n_state = torch.tensor(np.asarray(n_state, dtype=self.np_dtype), device=device)
        action_indices = torch.tensor(np.asarray(action), dtype=torch.long, device=device).unsqueeze(1)
        reward = np.asarray(reward, dtype=self.np_dtype)
        not_terminated = np.asarray(not_terminated, dtype=self.np_dtype)

        with torch.no_grad():
            n_oe_online, n_q_online = self.net.q_online(n_state)
            n_oe_target, n_q_target = self.net.q_target(n_state)

        if self.config.discount <= 0:
            with torch.no_grad():
                target_val = self.net.rnd_target(n_oe_target.detach())
            train_val = self.net.rnd_train(n_oe_target.detach())
            error = ((target_val - train_val) ** 2).mean(dim=1)

            loss_rnd = error.mean()
            self.opt_rnd.zero_grad()
            loss_rnd.backward()
            self.opt_rnd.step()
            self.trainer.info["loss_rnd"] = loss_rnd.item()

            error = error.detach().cpu().numpy()
            self.net.rnd_max = max(self.net.rnd_max, float(np.max(error)))
            discount = 1 - error / self.net.rnd_max
            self.trainer.info["discount"] = np.mean(discount)
        else:
            discount = self.config.discount

        # --- target_q
        n_q_online = n_q_online.detach().cpu().numpy()
        n_q_target = n_q_target.detach().cpu().numpy()
        n_act_idx = np.argmax(n_q_online, axis=1)
        if self.config.target_policy >= 1.0:
            n_q = n_q_target[np.arange(self.config.batch_size), n_act_idx]
        else:
            num_act = self.config.action_space.n
            w = np.full((self.config.batch_size, num_act), (1 - self.config.target_policy) / (num_act - 1), dtype=self.np_dtype)
            w[np.arange(self.config.batch_size), n_act_idx] = self.config.target_policy
            n_q = np.sum(n_q_target * w, axis=1)
        target_q_np = reward + not_terminated * discount * n_q
        target_q = torch.tensor(target_q_np, dtype=self.torch_dtype, device=device)

        # --- q
        oe_online, q_all = self.net.q_online(state)
        q = q_all.gather(1, action_indices).squeeze(1)
        loss_q = self.loss_q_func(target_q, q)
        if self.config.memory.requires_priority():
            loss_q = loss_q * weights
        loss_q = loss_q.mean()
        loss = loss_q
        self.trainer.info["loss_q"] = loss_q.item()

        # --- memory update
        if self.config.memory.requires_priority():
            q = q.detach().cpu().numpy()
            priorities = np.abs(target_q_np - q)
            self.memory.update_q(update_args, priorities, self.trainer.train_count)

        # --- feat
        if self.config.feat_type == "SimSiam":
            n_oe_hat = self.net.projector.trans(oe_online, action_indices.squeeze(-1))
            y_hat = self.net.projector.projection_and_head(n_oe_hat)

            with torch.no_grad():
                y_target = self.net.projector.projection(n_oe_target)

            loss_sim = self.loss_feat_func(y_target.detach(), y_hat)
            loss += loss_sim
            self.trainer.info["loss_sim"] = loss_sim.item()
        elif self.config.feat_type == "SPR":
            n_oe_hat = self.net.spr.trans(oe_online, action_indices.squeeze(-1))
            y_hat = self.net.spr.projection_and_head(n_oe_hat)

            with torch.no_grad():
                y_target = self.net.project_target_block(n_oe_target)

            loss_spr = self.loss_feat_func(y_target.detach(), y_hat)
            loss += loss_spr
            self.trainer.info["loss_spr"] = loss_spr.item()

        # --- latent
        if self.config.enable_archive:
            cat_oe = torch.cat([oe_online, n_oe_online], dim=0).detach()
            z = self.net.latent_encoder.encode(cat_oe)
            z = self.net.latent_encoder.decode(z)
            loss_lat = self.loss_lat_func(cat_oe, z)
            loss += loss_lat
            self.trainer.info["loss_ae"] = loss_lat.item()

        # --- bp
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.config.max_grad_norm)
        self.opt.step()
