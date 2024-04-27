import copy
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.rl.trainer import RLTrainer
from srl.rl import functions as funcs
from srl.rl.schedulers.scheduler import SchedulerConfig
from srl.rl.torch_.blocks.input_block import create_in_block_out_value

from .agent57_light import CommonInterfaceParameter, Config

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.input_ext_reward = config.input_ext_reward
        self.input_int_reward = config.input_int_reward
        self.input_action = config.input_action
        if not config.enable_intrinsic_reward:
            self.input_int_reward = False

        # --- in block
        self.input_block = create_in_block_out_value(
            config.input_value_block,
            config.input_image_block,
            config.observation_space,
        )

        # --- UVFA
        in_size = self.input_block.out_size
        if self.input_ext_reward:
            in_size += 1
        if self.input_int_reward:
            in_size += 1
        if self.input_action:
            in_size += config.action_space.n
        in_size += config.actor_num

        # out
        self.hidden_block = config.hidden_block.create_block_torch(
            in_size,
            config.action_space.n,
        )

    def forward(self, inputs):
        state = inputs[0]
        reward_ext = inputs[1]
        reward_int = inputs[2]
        onehot_action = inputs[3]
        onehot_actor = inputs[4]

        # input
        state = self.input_block(state)

        # UVFA
        uvfa_list = [state]
        if self.input_ext_reward:
            uvfa_list.append(reward_ext)
        if self.input_int_reward:
            uvfa_list.append(reward_int)
        if self.input_action:
            uvfa_list.append(onehot_action)
        uvfa_list.append(onehot_actor)
        x = torch.cat(uvfa_list, dim=1)

        x = self.hidden_block(x)
        return x


# ------------------------------------------------------
# エピソード記憶部(episodic_reward)
# ------------------------------------------------------
class _EmbeddingNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        # --- in block
        self.input_block = create_in_block_out_value(
            config.input_value_block,
            config.input_image_block,
            config.observation_space,
        )

        # --- emb
        self.emb_block = config.episodic_emb_block.create_block_torch(self.input_block.out_size)

        # --- out
        self.out_block = config.episodic_out_block.create_block_torch(self.emb_block.out_size * 2)
        self.out_block_normalize = nn.LayerNorm(self.out_block.out_size)
        self.out_block_out1 = nn.Linear(self.out_block.out_size, config.action_space.n)
        self.out_block_out2 = nn.Softmax(dim=1)

    def _image_call(self, state):
        x = self.input_block(state)
        return self.emb_block(x)

    def forward(self, x):
        x1 = self._image_call(x[0])
        x2 = self._image_call(x[1])

        x = torch.cat([x1, x2], dim=1)
        x = self.out_block(x)
        x = self.out_block_normalize(x)
        x = self.out_block_out1(x)
        x = self.out_block_out2(x)
        return x

    def predict(self, state):
        return self._image_call(state)


# ------------------------------------------------------
# 生涯記憶部(life long novelty module)
# ------------------------------------------------------
class _LifelongNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        # --- in block
        self.input_block = create_in_block_out_value(
            config.input_value_block,
            config.input_image_block,
            config.observation_space,
        )

        # hidden
        self.hidden_block = config.lifelong_hidden_block.create_block_torch(self.input_block.out_size)
        self.hidden_normalize = nn.LayerNorm(self.hidden_block.out_size)

    def forward(self, x):
        x = self.input_block(x)
        x = self.hidden_block(x)
        x = self.hidden_normalize(x)
        return x


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(CommonInterfaceParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.device = torch.device(self.config.used_device_torch)

        self.q_ext_online = QNetwork(self.config).to(self.device)
        self.q_ext_target = QNetwork(self.config).to(self.device)
        self.q_int_online = QNetwork(self.config).to(self.device)
        self.q_int_target = QNetwork(self.config).to(self.device)
        self.q_ext_target.eval()
        self.q_int_target.eval()
        self.q_ext_target.load_state_dict(self.q_ext_online.state_dict())
        self.q_int_target.load_state_dict(self.q_int_online.state_dict())

        self.emb_network = _EmbeddingNetwork(self.config).to(self.device)
        self.lifelong_target = _LifelongNetwork(self.config).to(self.device)
        self.lifelong_train = _LifelongNetwork(self.config).to(self.device)
        self.lifelong_target.eval()

    def call_restore(self, data: Any, from_cpu: bool = False, **kwargs) -> None:
        if self.config.used_device_torch != "cpu" and from_cpu:
            self.q_ext_online.to("cpu").load_state_dict(data[0])
            self.q_ext_target.to("cpu").load_state_dict(data[0])
            self.q_int_online.to("cpu").load_state_dict(data[1])
            self.q_int_target.to("cpu").load_state_dict(data[1])
            self.emb_network.to("cpu").load_state_dict(data[2])
            self.lifelong_target.to("cpu").load_state_dict(data[3])
            self.lifelong_train.to("cpu").load_state_dict(data[4])
            self.q_ext_online.to(self.device)
            self.q_ext_target.to(self.device)
            self.q_int_online.to(self.device)
            self.q_int_target.to(self.device)
            self.emb_network.to(self.device)
            self.lifelong_target.to(self.device)
            self.lifelong_train.to(self.device)
        else:
            self.q_ext_online.load_state_dict(data[0])
            self.q_ext_target.load_state_dict(data[0])
            self.q_int_online.load_state_dict(data[1])
            self.q_int_target.load_state_dict(data[1])
            self.emb_network.load_state_dict(data[2])
            self.lifelong_target.load_state_dict(data[3])
            self.lifelong_train.load_state_dict(data[4])

    def call_backup(self, to_cpu: bool = False, **kwargs):
        if self.config.used_device_torch != "cpu" and to_cpu:
            d = [
                copy.deepcopy(self.q_ext_online).to("cpu").state_dict(),
                copy.deepcopy(self.q_int_online).to("cpu").state_dict(),
                copy.deepcopy(self.emb_network).to("cpu").state_dict(),
                copy.deepcopy(self.lifelong_target).to("cpu").state_dict(),
                copy.deepcopy(self.lifelong_train).to("cpu").state_dict(),
            ]
        else:
            d = [
                self.q_ext_online.state_dict(),
                self.q_int_online.state_dict(),
                self.emb_network.state_dict(),
                self.lifelong_target.state_dict(),
                self.lifelong_train.state_dict(),
            ]
        return d

    def summary(self, **kwargs):
        print(self.q_ext_online)
        print(self.emb_network)
        print(self.lifelong_target)

    def predict_q_ext_online(self, x) -> np.ndarray:
        self.q_ext_online.eval()
        with torch.no_grad():
            q = self.q_ext_online(
                [
                    torch.tensor(x[0]).to(self.device),
                    torch.tensor(x[1]).to(self.device),
                    torch.tensor(x[2]).to(self.device),
                    torch.tensor(x[3]).to(self.device),
                    torch.tensor(x[4]).to(self.device),
                ]
            )
            q = q.to("cpu").detach().numpy()
        return q

    def predict_q_ext_target(self, x) -> np.ndarray:
        with torch.no_grad():
            q = self.q_ext_target(
                [
                    torch.tensor(x[0]).to(self.device),
                    torch.tensor(x[1]).to(self.device),
                    torch.tensor(x[2]).to(self.device),
                    torch.tensor(x[3]).to(self.device),
                    torch.tensor(x[4]).to(self.device),
                ]
            )
            q = q.to("cpu").detach().numpy()
        return q

    def predict_q_int_online(self, x) -> np.ndarray:
        self.q_int_online.eval()
        with torch.no_grad():
            q = self.q_int_online(
                [
                    torch.tensor(x[0]).to(self.device),
                    torch.tensor(x[1]).to(self.device),
                    torch.tensor(x[2]).to(self.device),
                    torch.tensor(x[3]).to(self.device),
                    torch.tensor(x[4]).to(self.device),
                ]
            )
            q = q.to("cpu").detach().numpy()
        return q

    def predict_q_int_target(self, x) -> np.ndarray:
        with torch.no_grad():
            q = self.q_int_target(
                [
                    torch.tensor(x[0]).to(self.device),
                    torch.tensor(x[1]).to(self.device),
                    torch.tensor(x[2]).to(self.device),
                    torch.tensor(x[3]).to(self.device),
                    torch.tensor(x[4]).to(self.device),
                ]
            )
            q = q.to("cpu").detach().numpy()
        return q

    def predict_emb(self, x) -> np.ndarray:
        self.emb_network.eval()
        with torch.no_grad():
            q = self.emb_network.predict(torch.tensor(x).to(self.device))
            q = q.to("cpu").detach().numpy()
        return q

    def predict_lifelong_target(self, x) -> np.ndarray:
        with torch.no_grad():
            q = self.lifelong_target(torch.tensor(x).to(self.device))
            q = q.to("cpu").detach().numpy()
        return q

    def predict_lifelong_train(self, x) -> np.ndarray:
        self.lifelong_train.eval()
        with torch.no_grad():
            q = self.lifelong_train(torch.tensor(x).to(self.device))
            q = q.to("cpu").detach().numpy()
        return q


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_sch_ext = SchedulerConfig.create_scheduler(self.config.lr_ext)
        self.lr_sch_int = SchedulerConfig.create_scheduler(self.config.lr_int)
        self.lr_sch_emb = SchedulerConfig.create_scheduler(self.config.episodic_lr)
        self.lr_sch_ll = SchedulerConfig.create_scheduler(self.config.lifelong_lr)

        self.q_ext_optimizer = optim.Adam(self.parameter.q_ext_online.parameters(), lr=self.lr_sch_ext.get_rate())
        self.q_int_optimizer = optim.Adam(self.parameter.q_int_online.parameters(), lr=self.lr_sch_int.get_rate())
        self.q_criterion = nn.HuberLoss()

        self.emb_optimizer = optim.Adam(self.parameter.emb_network.parameters(), lr=self.lr_sch_emb.get_rate())
        self.emb_criterion = nn.MSELoss()

        self.lifelong_optimizer = optim.Adam(self.parameter.lifelong_train.parameters(), lr=self.lr_sch_ll.get_rate())
        self.lifelong_criterion = nn.MSELoss()

        self.beta_list = funcs.create_beta_list(self.config.actor_num)
        self.discount_list = funcs.create_discount_list(self.config.actor_num)

        self.sync_count = 0

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        indices, batchs, weights = self.memory.sample(self.batch_size, self.train_count)

        device = self.parameter.device

        (
            states,
            n_states,
            onehot_actions,
            next_invalid_actions,
            next_invalid_actions_idx,
            rewards_ext,
            rewards_int,
            dones,
            prev_onehot_actions,
            prev_rewards_ext,
            prev_rewards_int,
            actor_idx,
            actor_idx_onehot,
        ) = self.parameter.change_batchs_format(batchs)

        batch_discount = np.array([self.discount_list[a] for a in actor_idx], np.float32)

        states = torch.from_numpy(states).to(device)
        n_states = torch.from_numpy(n_states).to(device)
        onehot_actions = torch.from_numpy(onehot_actions).to(device)

        # --- common params
        _params = [
            n_states,
            rewards_ext,
            rewards_int,
            onehot_actions,
            torch.from_numpy(actor_idx_onehot).to(device),
            next_invalid_actions_idx,
            next_invalid_actions,
            dones,
            batch_discount,
            #
            states,
            torch.from_numpy(prev_rewards_ext).to(device),
            torch.from_numpy(prev_rewards_int).to(device),
            torch.from_numpy(prev_onehot_actions).to(device),
            torch.from_numpy(weights).to(device),
            device,
        ]

        # --- update ext q
        self.parameter.q_ext_online.to(device)
        self.parameter.q_ext_target.to(device)
        td_errors_ext, ext_loss = self._update_q(
            True,
            self.parameter.q_ext_online,
            self.q_ext_optimizer,
            self.lr_sch_ext,
            rewards_ext,
            *_params,
        )
        self.info["ext_loss"] = ext_loss.item()

        # --- intrinsic reward
        if self.config.enable_intrinsic_reward:
            self.parameter.q_int_online.to(device)
            self.parameter.q_int_target.to(device)
            td_errors_int, int_loss = self._update_q(
                False,
                self.parameter.q_int_online,
                self.q_int_optimizer,
                self.lr_sch_int,
                rewards_int,
                *_params,
            )
            self.info["int_loss"] = int_loss.item()

            # ----------------------------------------
            # embedding network
            # ----------------------------------------
            self.parameter.emb_network.train()
            actions_probs = self.parameter.emb_network([states, n_states])
            emb_loss = self.emb_criterion(actions_probs, onehot_actions)

            self.emb_optimizer.zero_grad()
            emb_loss.backward()
            self.emb_optimizer.step()
            self.info["emb_loss"] = emb_loss.item()

            if self.lr_sch_emb.update(self.train_count):
                lr = self.lr_sch_emb.get_rate()
                for param_group in self.emb_optimizer.param_groups:
                    param_group["lr"] = lr

            # ----------------------------------------
            # lifelong network
            # ----------------------------------------
            with torch.no_grad():
                lifelong_target_val = self.parameter.lifelong_target(states)
            self.parameter.lifelong_train.train()
            lifelong_train_val = self.parameter.lifelong_train(states)
            lifelong_loss = self.lifelong_criterion(lifelong_target_val, lifelong_train_val)

            self.lifelong_optimizer.zero_grad()
            lifelong_loss.backward()
            self.lifelong_optimizer.step()
            self.info["lifelong_loss"] = lifelong_loss.item()

            if self.lr_sch_ll.update(self.train_count):
                lr = self.lr_sch_ll.get_rate()
                for param_group in self.lifelong_optimizer.param_groups:
                    param_group["lr"] = lr

        else:
            td_errors_int = 0.0

        # --- update memory
        if self.config.disable_int_priority:
            priorities = np.abs(td_errors_ext)
        else:
            batch_beta = np.array([self.beta_list[a] for a in actor_idx], np.float32)
            priorities = np.abs(td_errors_ext + batch_beta * td_errors_int)
        self.memory.update(indices, batchs, priorities)

        # --- sync target
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_ext_target.load_state_dict(self.parameter.q_ext_online.state_dict())
            self.parameter.q_int_target.load_state_dict(self.parameter.q_int_online.state_dict())
            self.sync_count += 1

        self.info["sync"] = self.sync_count
        self.train_count += 1

    def _update_q(
        self,
        is_ext: bool,
        model_q_online,
        optimizer,
        lr_sch,
        rewards,  # (batch, 1)
        #
        n_states,
        rewards_ext,  # (batch, 1)
        rewards_int,  # (batch, 1)
        actions_onehot,
        actor_idx_onehot,
        next_invalid_actions_idx,
        next_invalid_actions,
        dones,  # (batch)
        batch_discount,  # (batch)
        #
        states,
        prev_rewards_ext,  # (batch, 1)
        prev_rewards_int,  # (batch, 1)
        prev_actions_onehot,
        weights,  # (batch)
        device,
    ):
        target_q = self.parameter.calc_target_q(
            is_ext,
            rewards.reshape(-1),
            #
            n_states,
            rewards_ext,
            rewards_int,
            actions_onehot,
            actor_idx_onehot,
            next_invalid_actions_idx,
            next_invalid_actions,
            dones,
            batch_discount,
        )
        target_q = torch.from_numpy(target_q).to(device)

        # --- train
        model_q_online.train()
        q = model_q_online(
            [
                states,
                prev_rewards_ext,
                prev_rewards_int,
                prev_actions_onehot,
                actor_idx_onehot,
            ]
        )
        q = torch.sum(q * actions_onehot, dim=1)  # (batch, shape) -> (batch)

        loss = self.q_criterion(target_q * weights, q * weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_sch.update(self.train_count):
            lr = lr_sch.get_rate()
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        td_errors = (target_q - q).to("cpu").detach().numpy()
        return td_errors, loss
