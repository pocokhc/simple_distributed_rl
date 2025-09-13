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

    def train(self):
        batches = self.memory.sample()
        if batches is None:
            return
        batches, weights, update_args = batches
        if self.config.memory.requires_priority():
            weights = torch.tensor(np.asarray(weights, dtype=self.np_dtype), device=self.device)
        else:
            weights = 1

        state, n_state, action, reward, not_done, total_reward = zip(*batches)
        state = torch.tensor(np.asarray(state, dtype=self.np_dtype), device=self.device)
        n_state = torch.tensor(np.asarray(n_state, dtype=self.np_dtype), device=self.device)
        action_indices = torch.tensor(np.asarray(action), dtype=torch.long, device=self.device).unsqueeze(1)
        reward = torch.tensor(np.array(reward, dtype=self.np_dtype), device=self.device)
        not_done = torch.tensor(np.array(not_done, dtype=self.np_dtype), device=self.device)
        total_reward = torch.tensor(np.array(total_reward, dtype=self.np_dtype), device=self.device)

        loss = 0

        q_all = self.parameter.q_online(torch.cat([state, n_state]))
        q = q_all[: self.config.batch_size]
        n_q = q_all[self.config.batch_size :].detach()

        n_maxq = n_q.max(dim=1).values
        target_q = reward + not_done * self.config.discount * n_maxq

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
