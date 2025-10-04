from itertools import chain
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.rl.trainer import RLTrainer
from srl.rl.torch_.functions import inverse_linear_symlog, linear_symlog
from srl.rl.torch_.helper import model_soft_sync

from .config import Config
from .memory import Memory
from .torch_model import Model


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
        self.info = trainer.info
        self.config: Config = trainer.config
        self.memory: Memory = trainer.memory
        self.net: Model = trainer.parameter.net
        self.torch_dtype = self.config.get_dtype("torch")
        self.np_dtype = self.config.get_dtype("np")
        self.act_num = self.config.action_space.n

        self.train_count = 0

        # --- reset params
        self.reset_params = []

        # --- q
        self.models = [self.net.encoder, self.net.q_online]
        self.loss_q_func = nn.HuberLoss(reduction="none")
        self.loss_align_func = nn.MSELoss(reduction="none")
        self.reset_params += list(self.net.encoder.parameters())
        self.reset_params += list(self.net.q_online.parameters())

        if self.config.feat_type == "BYOL":
            self.models.append(self.net.byol_online)
            self.reset_params += list(self.net.byol_online.parameters())

        if self.config.enable_int_q:
            self.models.append(self.net.q_int_online)
            self.reset_params += list(self.net.q_int_online.parameters())
            self.loss_int_q_func = nn.HuberLoss(reduction="none")
            self.loss_int_align_func = nn.MSELoss(reduction="none")

            # --- rnd
            if self.config.feat_type == "":
                self.opt_rnd = optim.Adam(self.net.rnd.parameters(), lr=self.config.lr / 5)
                self.reset_params += list(self.net.rnd.parameters())

            # --- episodic
            if self.config.enable_int_episodic:
                self.models.append(self.net.emb_net)
                self.reset_params += list(self.net.emb_net.parameters())
                self.loss_emb_func = nn.CrossEntropyLoss()

        self.params = list(chain(*[m.parameters() for m in self.models]))
        if self.config.replay_ratio > 1:
            self.opt = optim.AdamW(self.params, lr=self.config.lr, weight_decay=0.1)
        else:
            self.opt = optim.RAdam(self.params, lr=self.config.lr)
        [m.train() for m in self.models]

        self.states_np_list = []
        for space in self.config.observation_space.spaces:
            self.states_np_list.append(
                np.empty(
                    (self.config.batch_size * 2, *space.shape),
                    dtype=space.dtype,
                )
            )
        self.action_indices_np = np.empty((self.config.batch_size, 1), dtype=np.int64)
        self.reward_np = np.empty((self.config.batch_size,), dtype=self.np_dtype)
        self.not_terminated_np = np.empty((self.config.batch_size,), dtype=self.np_dtype)
        self.total_reward_np = np.empty((self.config.batch_size,), dtype=self.np_dtype)

        self.reset_net = 0

    def train(self):
        for _ in range(self.config.replay_ratio):
            batches = self.memory.sample_q()
            if batches is None:
                break

            # --- reset
            if (self.config.reset_net_interval > 0) and (self.train_count % self.config.reset_net_interval == 1):
                reset_model_params(self.reset_params, (1 - self.config.lr), self.config.lr, stddev=0.1)
                self.reset_net += 1
                self.info["reset_net"] = self.reset_net

            # --- train
            self._train(batches)

            # --- target sync
            if self.config.feat_type == "BYOL":
                if self.train_count % self.config.byol_model_update_interval == 0:
                    model_soft_sync(self.net.byol_target, self.net.byol_online.proj_block, self.config.byol_model_update_rate)

            self.train_count += 1

    def _train(self, batches):
        device = self.net.device
        batches, weights, update_args = batches

        for i, b in enumerate(batches):
            for j in range(self.config.observation_space.space_size):
                self.states_np_list[j][i] = b[0][j]
                self.states_np_list[j][self.config.batch_size + i] = b[1][j]
            self.action_indices_np[i] = b[2]
            self.reward_np[i] = b[3]
            self.not_terminated_np[i] = b[4]
            self.total_reward_np[i] = b[5]
        states_list = [
            torch.from_numpy(self.states_np_list[j]).to(device)
            for j in range(self.config.observation_space.space_size)  #
        ]
        action_indices = torch.from_numpy(self.action_indices_np).to(device)
        reward = torch.from_numpy(self.reward_np).to(device)
        not_terminated = torch.from_numpy(self.not_terminated_np).to(device)
        total_reward = torch.from_numpy(self.total_reward_np).to(device)

        if self.config.memory.requires_priority():
            weights = torch.tensor(np.asarray(weights), dtype=self.torch_dtype, device=device)
        else:
            weights = 1

        loss = 0
        oe_s = self.net.encoder(states_list)
        oe = oe_s[: self.config.batch_size]
        n_oe = oe_s[self.config.batch_size :]
        q_acts_s, v_s = self.net.q_online(oe_s)

        # --- rnd
        if self.config.enable_int_q and self.config.feat_type == "":
            rnd_error = self.net.rnd.compute_intrinsic_reward(n_oe.detach(), update=True, norm=False)
            # rnd_error = self.net.rnd.compute_intrinsic_reward(oe_s.detach(), update=True, norm=False)
            # loss_rnd = rnd_error[: self.config.batch_size].mean()
            loss_rnd = rnd_error.mean()
            self.info["loss_rnd"] = loss_rnd.item()
            self.info["rnd_min"] = self.net.rnd.error_norm.get_min()
            self.info["rnd_var"] = self.net.rnd.error_norm.get_var()

            self.opt_rnd.zero_grad()
            loss_rnd.backward()
            self.opt_rnd.step()

            rnd_error = rnd_error.detach()

        # --- target_q
        n_q = q_acts_s[self.config.batch_size :].detach().max(dim=1).values
        if self.config.enable_q_rescale:
            n_q = inverse_linear_symlog(n_q)
        target_q = reward + not_terminated * self.config.discount * n_q
        if self.config.enable_q_rescale:
            target_q = linear_symlog(target_q)

        # --- q
        q = q_acts_s[: self.config.batch_size].gather(1, action_indices).squeeze(1)
        loss_q = (self.loss_q_func(target_q, q) * weights).mean()
        loss += loss_q
        self.info["loss_q"] = loss_q.item()

        # --- alignment q
        if self.config.enable_q_rescale:
            total_reward = linear_symlog(total_reward)
        loss_align = (self.loss_align_func(total_reward, q) * weights).mean()
        loss += self.config.align_loss_coeff * loss_align
        self.info["loss_align"] = loss_align.item()

        # --- memory update
        if self.config.memory.requires_priority():
            priorities = np.abs((target_q - q).detach().cpu().numpy())
            self.memory.update_q(update_args, priorities, self.train_count)

        # --- feat
        if self.config.feat_type == "BYOL":
            oe = oe_s[: self.config.batch_size]
            y_hat = self.net.byol_online(oe, action_indices.squeeze(-1))
            with torch.no_grad():
                n_oe = oe_s[self.config.batch_size :].detach()
                y_target = self.net.byol_target(n_oe)
            loss_byol, int_rew = self.net.byol_online.compute_loss_and_reward(y_target.detach(), y_hat)
            loss_byol = (loss_byol * weights).mean()
            loss += loss_byol
            self.info["loss_byol"] = loss_byol.item()

        if self.config.enable_int_q:
            if self.config.feat_type == "":  # --- RND
                int_rew = self.net.rnd.norm(rnd_error)
            elif self.config.feat_type == "BYOL":
                int_rew = int_rew.detach()
                self.info["byol_min"] = self.net.byol_online.reward_norm.get_min()
                self.info["byol_var"] = self.net.byol_online.reward_norm.get_var()
            self.info["int_reward"] = int_rew.mean().item()

            # --- q int target
            q_int_acts_s, v_int_acts_s = self.net.q_int_online(oe_s.detach())
            n_q_int = q_int_acts_s[self.config.batch_size :].detach()
            n_q_int = torch.max(n_q_int, dim=-1).values
            n_v_int = v_int_acts_s[self.config.batch_size :].detach().squeeze(-1)
            target_q_int = int_rew + not_terminated * self.config.int_discount * (n_q_int + n_v_int) / 2

            # --- q int train
            q_int = q_int_acts_s[: self.config.batch_size].gather(1, action_indices).squeeze(1)
            loss_int_q = (self.loss_int_q_func(target_q_int, q_int) * weights).mean()
            loss += loss_int_q
            self.info["loss_int_q"] = loss_int_q.item()

            # --- alignment q int
            loss_int_align = (self.loss_int_align_func(int_rew, q_int) * weights).mean()
            loss += self.config.int_align_loss_coeff * loss_int_align
            self.info["loss_int_align"] = loss_int_align.item()

            # --- int emb
            if self.config.enable_int_episodic:
                act_logits = self.net.emb_net(oe.detach(), n_oe.detach())
                a = torch.nn.functional.one_hot(action_indices.squeeze(-1), self.config.action_space.n).float()
                loss_emb = self.loss_emb_func(act_logits, a)
                loss += loss_emb
                self.info["loss_emb"] = loss_emb.item()

        # --- bp
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.config.max_grad_norm)
        self.opt.step()
