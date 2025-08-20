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

        self.alignment_loss_coeff_sch = self.config.alignment_loss_coeff_scheduler.create(self.config.alignment_loss_coeff)

    def train(self):
        batches = self.memory.sample()
        if batches is None:
            return
        batches, weights, update_args = batches

        state, n_state, action, reward, not_done, discounted_reward = zip(*batches)
        state = torch.tensor(np.asarray(state, dtype=self.np_dtype), device=self.device)
        n_state = torch.tensor(np.asarray(n_state, dtype=self.np_dtype), device=self.device)
        action_indices = torch.tensor(np.asarray(action), dtype=torch.long, device=self.device).unsqueeze(1)
        reward = torch.tensor(np.array(reward, dtype=self.np_dtype), device=self.device)
        not_done = torch.tensor(np.array(not_done, dtype=self.np_dtype), device=self.device)
        discounted_reward = torch.tensor(np.array(discounted_reward, dtype=self.np_dtype), device=self.device)

        with torch.no_grad():
            n_q = self.parameter.q_online(n_state)
        n_maxq = n_q.max(dim=1).values
        target_q = reward + not_done * self.config.discount * n_maxq

        q = self.parameter.q_online(state)
        q = q.gather(1, action_indices).squeeze(1)
        loss = self.criterion1(target_q, q)
        if self.config.memory.requires_priority():
            weights = torch.tensor(np.asarray(weights, dtype=self.np_dtype), device=self.device)
            loss = loss * weights
        loss = loss.mean()
        self.info["loss"] = loss.item()

        loss_alignment = self.criterion2(discounted_reward, q)
        if self.config.memory.requires_priority():
            loss_alignment = loss_alignment * weights
        loss_alignment = loss_alignment.mean()
        align_coeff = self.alignment_loss_coeff_sch.update(self.train_count).to_float()
        loss += align_coeff * loss_alignment
        self.info["loss_align"] = loss_alignment.item()
        self.info["loss_align_coeff"] = align_coeff

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # --- update
        if self.config.memory.requires_priority():
            priorities = np.abs((target_q - q).detach().cpu().numpy())
            self.memory.update(update_args, priorities, self.train_count)

        self.train_count += 1
