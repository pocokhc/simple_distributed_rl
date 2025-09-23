import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.rl.trainer import RLTrainer

from .config import Config, Memory
from .parameter import Parameter

logger = logging.getLogger(__name__)


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self):
        self.np_dtype = self.config.get_dtype("np")
        self.device = self.parameter.device

        self.opt = optim.Adam(self.parameter.q_online.parameters(), lr=self.config.lr)
        self.criterion1 = nn.HuberLoss(reduction="none")
        self.criterion2 = nn.MSELoss(reduction="none")

        self.align_loss_coeff_sch = self.config.align_loss_coeff_scheduler.create(self.config.align_loss_coeff)

        self.states_np = np.empty((self.config.batch_size * 2, *self.config.observation_space.shape), dtype=self.np_dtype)
        self.action_indices_np = np.empty((self.config.batch_size, 1), dtype=np.int64)
        self.reward_np = np.empty((self.config.batch_size,), dtype=self.np_dtype)
        self.not_terminated_np = np.empty((self.config.batch_size,), dtype=self.np_dtype)
        self.total_reward_np = np.empty((self.config.batch_size,), dtype=self.np_dtype)

    def train(self):
        batches = self.memory.sample()
        if batches is None:
            return
        batches, weights, update_args = batches
        if self.config.memory.requires_priority():
            weights = torch.tensor(np.asarray(weights, dtype=self.np_dtype), device=self.device)
        else:
            weights = 1

        for i, b in enumerate(batches):
            self.states_np[i] = b[0]
            self.states_np[self.config.batch_size + i] = b[1]
            self.action_indices_np[i] = b[2]
            self.reward_np[i] = b[3]
            self.not_terminated_np[i] = b[4]
            self.total_reward_np[i] = b[5]
        states = torch.from_numpy(self.states_np).to(self.device)
        action_indices = torch.from_numpy(self.action_indices_np).to(self.device)
        reward = torch.from_numpy(self.reward_np).to(self.device)
        not_terminated = torch.from_numpy(self.not_terminated_np).to(self.device)
        total_reward = torch.from_numpy(self.total_reward_np).to(self.device)

        loss = 0

        q_all, v_all = self.parameter.q_online(states)
        q = q_all[: self.config.batch_size]
        n_q = q_all[self.config.batch_size :].detach()
        n_v = v_all[self.config.batch_size :].detach().squeeze(-1)

        n_maxq = n_q.max(dim=1).values
        target_q = reward + not_terminated * self.config.discount * (n_maxq + n_v) / 2

        q = q.gather(1, action_indices).squeeze(1)
        loss_q = (self.criterion1(target_q, q) * weights).mean()
        loss += loss_q
        self.info["loss_q"] = loss_q.item()

        loss_alignment = (self.criterion2(total_reward, q) * weights).mean()
        align_coeff = self.align_loss_coeff_sch.update(self.train_count).to_float()
        loss += align_coeff * loss_alignment
        self.info["loss_align"] = loss_alignment.item()
        if self.config.align_loss_coeff_scheduler.is_update_step():
            self.info["loss_align_coeff"] = align_coeff

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # --- update
        if self.config.memory.requires_priority():
            priorities = np.abs((target_q - q).detach().cpu().numpy())
            self.memory.update(update_args, priorities, self.train_count)

        self.train_count += 1
