import logging
from itertools import chain
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.rl.trainer import RLTrainer
from srl.rl.torch_.helper import model_soft_sync

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
        self.act_num = self.config.action_space.n

        # --- reset params
        self.reset_params = []

        # --- q
        self.models = [self.net.q_online]
        self.loss_q_func = nn.HuberLoss(reduction="none")
        self.loss_align_func = nn.MSELoss(reduction="none")
        self.reset_params += list(self.net.q_online.parameters())

        if self.config.feat_type == "SimSiam":
            self.models.append(self.net.projector)
            self.reset_params += list(self.net.projector.parameters())
        elif self.config.feat_type == "BYOL":
            self.models.append(self.net.byol_online)
            self.reset_params += list(self.net.byol_online.parameters())

        if self.config.enable_int_q:
            self.models.append(self.net.q_int_online)
            self.reset_params += list(self.net.q_int_online.parameters())
            self.loss_int_q_func = nn.HuberLoss(reduction="none")
            self.loss_int_align_func = nn.MSELoss(reduction="none")

        self.params = list(chain(*[m.parameters() for m in self.models]))
        if self.config.replay_ratio > 1:
            self.opt = optim.AdamW(self.params, lr=self.config.lr, weight_decay=0.1)
        else:
            self.opt = optim.RAdam(self.params, lr=self.config.lr)
        [m.train() for m in self.models]

        self.reset_net = 0

    def train(self):
        for _ in range(self.config.replay_ratio):
            batches = self.memory.sample_q()
            if batches is None:
                break

            # --- reset
            if (self.config.reset_net_interval > 0) and (self.trainer.train_count % self.config.reset_net_interval == 1):
                reset_model_params(self.reset_params, (1 - self.config.lr), self.config.lr, stddev=0.1)
                self.reset_net += 1
                self.trainer.info["reset_net"] = self.reset_net

            # --- train
            self._train(batches)

            # --- target sync
            if self.config.feat_type == "BYOL":
                if self.trainer.train_count % self.config.byol_model_update_interval == 0:
                    model_soft_sync(self.net.byol_target, self.net.byol_online.proj_block, self.config.byol_model_update_rate)

            self.trainer.train_count += 1

    def _train(self, batches):
        device = self.net.device
        batches, weights, update_args = batches

        state, n_state, action, reward, not_terminated, total_reward = zip(*batches)
        state = torch.tensor(np.asarray(state, dtype=self.np_dtype), device=device)
        n_state = torch.tensor(np.asarray(n_state, dtype=self.np_dtype), device=device)
        action_indices = torch.tensor(np.asarray(action), dtype=torch.long, device=device).unsqueeze(1)
        reward = torch.tensor(np.asarray(reward, dtype=self.np_dtype), device=device)
        not_terminated = torch.tensor(np.asarray(not_terminated, dtype=self.np_dtype), device=device)
        total_reward = torch.tensor(np.asarray(total_reward, dtype=self.np_dtype), device=device)
        if self.config.memory.requires_priority():
            weights = torch.tensor(np.asarray(weights), dtype=self.torch_dtype, device=device)
        else:
            weights = 1

        loss = 0
        with torch.no_grad():
            n_oe_online, n_q_online = self.net.q_online(n_state)
        oe_online, q_all = self.net.q_online(state)

        # --- target_q
        n_q = n_q_online.detach().max(dim=1).values
        target_q = reward + not_terminated * self.config.discount * n_q

        # --- q
        q = q_all.gather(1, action_indices).squeeze(1)
        loss_q = (self.loss_q_func(target_q, q) * weights).mean()
        loss += loss_q
        self.trainer.info["loss_q"] = loss_q.item()

        # --- alignment q
        loss_align = (self.loss_align_func(total_reward, q) * weights).mean()
        loss += self.config.align_loss_coeff * loss_align
        self.trainer.info["loss_align"] = loss_align.item()

        # --- memory update
        if self.config.memory.requires_priority():
            priorities = np.abs((target_q - q).detach().cpu().numpy())
            self.memory.update_q(update_args, priorities, self.trainer.train_count)

        # --- feat
        if self.config.feat_type == "SimSiam":
            y_hat = self.net.projector(oe_online, action_indices.squeeze(-1))
            with torch.no_grad():
                y_target = self.net.projector.projection(n_oe_online)
            loss_sim, int_rew = self.net.projector.compute_loss_and_reward(y_target.detach(), y_hat)
            loss_sim = (loss_sim * weights).mean()
            loss += loss_sim
            self.trainer.info["loss_sim"] = loss_sim.item()
        elif self.config.feat_type == "BYOL":
            y_hat = self.net.byol_online(oe_online, action_indices.squeeze(-1))
            with torch.no_grad():
                y_target = self.net.byol_target(n_oe_online)
            loss_byol, int_rew = self.net.byol_online.compute_loss_and_reward(y_target.detach(), y_hat)
            loss_byol = (loss_byol * weights).mean()
            loss += loss_byol
            self.trainer.info["loss_byol"] = loss_byol.item()

        if self.config.enable_int_q:
            # --- q int target
            with torch.no_grad():
                n_q_int_online = self.net.q_int_online(n_oe_online)
            w = torch.full(
                (self.config.batch_size, self.act_num),
                (1 - self.config.int_target_prob) / (self.act_num - 1),
                dtype=self.torch_dtype,
                device=device,
            )
            n_int_act_idx = torch.argmax(n_q_int_online, dim=1)
            w.scatter_(1, n_int_act_idx.unsqueeze(1), self.config.int_target_prob)
            n_q_int = (n_q_int_online * w).sum(dim=1)
            target_q_int = int_rew + not_terminated * self.config.int_discount * n_q_int

            # --- q int train
            q_int_all = self.net.q_int_online(oe_online.detach())
            q_int = q_int_all.gather(1, action_indices).squeeze(1)
            loss_int_q = (self.loss_int_q_func(target_q_int, q_int) * weights).mean()
            loss += loss_int_q
            self.trainer.info["loss_int_q"] = loss_int_q.item()

            # --- alignment q int
            loss_int_align = (self.loss_int_align_func(int_rew, q_int) * weights).mean()
            loss += self.config.int_align_loss_coeff * loss_int_align
            self.trainer.info["loss_int_align"] = loss_int_align.item()

        # --- bp
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.config.max_grad_norm)
        self.opt.step()
