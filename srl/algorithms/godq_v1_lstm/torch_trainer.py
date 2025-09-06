import logging
from itertools import chain
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.rl.trainer import RLTrainer
from srl.rl.torch_.helper import decode_sequence_batch, encode_sequence_batch, model_soft_sync

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
        self.models = [self.net.encoder, self.net.q_online]
        self.loss_q_func = nn.HuberLoss(reduction="none")
        self.loss_align_func = nn.MSELoss(reduction="none")
        self.reset_params += list(self.net.encoder.parameters())
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
        lr2 = self.config.lr * ((self.config.batch_size // 2) * (self.config.batch_length + 1)) / (self.config.batch_size * self.config.batch_length)
        if self.config.replay_ratio > 1:
            self.opt = optim.AdamW(self.params, lr=self.config.lr, weight_decay=0.1)
            self.opt2 = optim.AdamW(self.params, lr=lr2, weight_decay=0.1)
        else:
            self.opt = optim.RAdam(self.params, lr=self.config.lr)
            self.opt2 = optim.RAdam(self.params, lr=lr2)
        [m.train() for m in self.models]

        self.int_target_w = torch.full(
            (self.config.batch_size, self.config.batch_length + 1, self.act_num),
            (1 - self.config.int_target_prob) / (self.act_num - 1),
            dtype=self.torch_dtype,
            device=self.net.device,
        )
        self.hc = self.net.encoder.get_initial_state(self.config.batch_size)

        self.states_np = np.empty((self.config.batch_size, self.config.batch_length + 2, *self.config.observation_space.shape), dtype=self.np_dtype)
        self.action_indices_np = np.empty((self.config.batch_size, self.config.batch_length + 2), dtype=np.int64)
        self.rewards_np = np.empty((self.config.batch_size, self.config.batch_length + 2), dtype=self.np_dtype)
        self.not_terminateds_np = np.empty((self.config.batch_size, self.config.batch_length + 2), dtype=self.np_dtype)
        self.not_starts_np = np.empty((self.config.batch_size, self.config.batch_length + 2), dtype=np.int8)
        self.not_dones_np = np.empty((self.config.batch_size, self.config.batch_length + 2), dtype=np.bool)
        self.total_rewards_np = np.empty((self.config.batch_size, self.config.batch_length + 2), dtype=self.np_dtype)

        self.reset_net = 0

    def train(self):
        for _ in range(self.config.replay_ratio):
            batches = self.memory.sample_sequential()
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
        batch_size = len(batches)
        batch_length = len(batches[0])

        # print([[b[6] for b in steps] for steps in batches])  # debug

        for i, steps in enumerate(batches):
            for j, b in enumerate(steps):
                self.states_np[i, j] = b[0]
                self.action_indices_np[i, j] = b[1]
                self.rewards_np[i, j] = b[2]
                self.not_terminateds_np[i, j] = b[3]
                self.not_starts_np[i, j] = b[4]
                self.not_dones_np[i, j] = b[5]
                self.total_rewards_np[i, j] = b[7]
        states = torch.from_numpy(self.states_np[:batch_size, :batch_length, ...]).to(device)
        action_indices = torch.from_numpy(self.action_indices_np[:batch_size, :batch_length, ...]).to(device)
        rewards = torch.from_numpy(self.rewards_np[:batch_size, :batch_length, ...]).to(device)
        not_terminateds = torch.from_numpy(self.not_terminateds_np[:batch_size, :batch_length, ...]).to(device)
        not_starts = torch.from_numpy(self.not_starts_np[:batch_size, :batch_length, ...]).unsqueeze(-1).to(device)
        not_dones = torch.from_numpy(self.not_dones_np[:batch_size, :batch_length, ...]).to(device)
        total_rewards = torch.from_numpy(self.total_rewards_np[:batch_size, :batch_length, ...]).to(device)

        loss = 0

        # --- state/act encode
        ts, head_size1, head_size2 = encode_sequence_batch(states)
        ta, head_size1, head_size2 = encode_sequence_batch(action_indices)
        enc_sa_s = self.net.encoder.forward_encode(ts, ta)
        enc_sa_s = decode_sequence_batch(enc_sa_s, head_size1, head_size2)

        # --- lstm
        z_list = []
        hc = (self.hc[0][:batch_size].detach(), self.hc[1][:batch_size].detach())
        for i in range(batch_length):
            hc = (hc[0] * not_starts[:, i, ...], hc[1] * not_starts[:, i, ...])
            z, hc = self.net.encoder.forward_lstm(enc_sa_s[:, i, ...], hc)
            z_list.append(z)
        z_s = torch.stack(z_list, dim=0)
        z_s = torch.permute(z_s, (1, 0, 2))
        self.hc[0][:batch_size] = hc[0]
        self.hc[1][:batch_size] = hc[1]

        # --- q
        q_all_s = self.net.q_online(z_s)
        q_all = q_all_s[:, :-1, ...]
        n_q_all = q_all_s[:, 1:, ...]

        n_q = n_q_all.detach().max(dim=-1).values
        target_q = rewards[:, 1:, ...] + not_terminateds[:, 1:, ...] * self.config.discount * n_q

        q = q_all.gather(2, action_indices[:, 1:].unsqueeze(-1)).squeeze(-1)
        loss_q = self.loss_q_func(target_q, q)
        loss += loss_q
        self.trainer.info["loss_q"] = loss_q.mean().item()

        loss_align = self.loss_align_func(total_rewards[:, 1:, ...], q)
        loss += self.config.align_loss_coeff * loss_align
        self.trainer.info["loss_align"] = loss_align.mean().item()

        # --- feat
        if self.config.feat_type == "SimSiam":
            z, head_size1, head_size2 = encode_sequence_batch(z_s[:, :-1, ...])
            n_z, head_size1, head_size2 = encode_sequence_batch(z_s[:, 1:, ...])
            a, head_size1, head_size2 = encode_sequence_batch(action_indices[:, 1:])
            y_hat = self.net.projector(z, a)
            with torch.no_grad():
                y_target = self.net.projector.projection(n_z.detach())
            loss_sim, int_rew = self.net.projector.compute_loss_and_reward(y_target.detach(), y_hat)
            int_rew = decode_sequence_batch(int_rew, head_size1, head_size2)
            loss_sim = decode_sequence_batch(loss_sim, head_size1, head_size2)
            loss += loss_sim
            self.trainer.info["loss_sim"] = loss_sim.mean().item()
        elif self.config.feat_type == "BYOL":
            z, head_size1, head_size2 = encode_sequence_batch(z_s[:, :-1, ...])
            n_z, head_size1, head_size2 = encode_sequence_batch(z_s[:, 1:, ...])
            a, head_size1, head_size2 = encode_sequence_batch(action_indices[:, 1:])
            y_hat = self.net.byol_online(z, a)
            with torch.no_grad():
                y_target = self.net.byol_target(n_z.detach())
            loss_byol, int_rew = self.net.byol_online.compute_loss_and_reward(y_target.detach(), y_hat)
            int_rew = decode_sequence_batch(int_rew, head_size1, head_size2)
            loss_byol = decode_sequence_batch(loss_byol, head_size1, head_size2)
            loss += loss_byol
            self.trainer.info["loss_byol"] = loss_byol.mean().item()

        # --- int q
        if self.config.enable_int_q:
            q_int_all_s = self.net.q_int_online(z_s.detach())
            q_int_all = q_int_all_s[:, :-1, ...]
            n_q_int_all = q_int_all_s[:, 1:, ...].detach()

            n_int_act_idx = torch.argmax(n_q_int_all, dim=2)
            w = self.int_target_w[:batch_size, : batch_length - 1]
            w.fill_((1 - self.config.int_target_prob) / (self.act_num - 1))
            w.scatter_(2, n_int_act_idx.unsqueeze(-1), self.config.int_target_prob)
            n_q_int = (n_q_int_all * w).sum(dim=2)
            target_q_int = int_rew + not_terminateds[:, 1:, ...] * self.config.int_discount * n_q_int

            q_int = q_int_all.gather(2, action_indices[:, 1:].unsqueeze(-1)).squeeze(-1)
            loss_int_q = self.loss_int_q_func(target_q_int, q_int)
            loss += loss_int_q
            self.trainer.info["loss_int_q"] = loss_int_q.mean().item()

            loss_int_align = self.loss_int_align_func(int_rew, q_int)
            loss += self.config.int_align_loss_coeff * loss_int_align
            self.trainer.info["loss_int_align"] = loss_int_align.mean().item()

        # 今が終了、次が開始の境目は学習しない
        _masked_loss = loss[not_dones[:, :-1]]
        loss = _masked_loss.mean() if _masked_loss.numel() > 0 else torch.tensor(0.0, device=_masked_loss.device, requires_grad=True)

        # --- bp
        opt = self.opt if self.config.batch_size == batch_size else self.opt2
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.config.max_grad_norm)
        opt.step()
