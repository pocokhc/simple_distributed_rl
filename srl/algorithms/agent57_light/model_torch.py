import copy
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.rl.trainer import RLTrainer
from srl.rl import functions as funcs
from srl.rl.torch_.helper import model_backup, model_restore

from .agent57_light import CommonInterfaceParameter, Config, Memory

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.input_ext_reward = config.input_ext_reward
        self.input_int_reward = config.input_int_reward
        self.input_action = config.input_action
        if not config.enable_intrinsic_reward:
            self.input_int_reward = False

        if config.observation_space.is_value():
            self.in_block = config.input_value_block.create_torch_block(config.observation_space.shape)
        elif config.observation_space.is_image():
            self.in_block = config.input_image_block.create_torch_block(config.observation_space)
        else:
            raise ValueError(config.observation_space)

        # --- UVFA
        in_size = self.in_block.out_size
        if self.input_ext_reward:
            in_size += 1
        if self.input_int_reward:
            in_size += 1
        if self.input_action:
            in_size += config.action_space.n
        in_size += config.actor_num

        # out
        self.hidden_block = config.hidden_block.create_torch_block(in_size, config.action_space.n)

    def forward(self, inputs):
        state = inputs[0]
        reward_ext = inputs[1]
        reward_int = inputs[2]
        onehot_action = inputs[3]
        onehot_actor = inputs[4]

        # input
        state = self.in_block(state)

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

        if config.observation_space.is_value():
            self.in_block = config.input_value_block.create_torch_block(config.observation_space.shape)
        elif config.observation_space.is_image():
            self.in_block = config.input_image_block.create_torch_block(config.observation_space)
        else:
            raise ValueError(config.observation_space)

        self.emb_block = config.episodic_emb_block.create_torch_block(self.in_block.out_size)

        # --- out
        self.out_block = config.episodic_out_block.create_torch_block(self.emb_block.out_size * 2)
        self.out_block_normalize = nn.LayerNorm(self.out_block.out_size)
        self.out_block_out1 = nn.Linear(self.out_block.out_size, config.action_space.n)
        self.out_block_out2 = nn.Softmax(dim=1)

    def _image_call(self, state):
        x = self.in_block(state)
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

        if config.observation_space.is_value():
            self.in_block = config.input_value_block.create_torch_block(config.observation_space.shape)
        elif config.observation_space.is_image():
            self.in_block = config.input_image_block.create_torch_block(config.observation_space)
        else:
            raise ValueError(config.observation_space)

        self.hidden_block = config.lifelong_hidden_block.create_torch_block(self.in_block.out_size)
        self.hidden_normalize = nn.LayerNorm(self.hidden_block.out_size)

    def forward(self, x):
        x = self.in_block(x)
        x = self.hidden_block(x)
        x = self.hidden_normalize(x)
        return x


class Parameter(CommonInterfaceParameter):
    def setup(self):
        super().setup()
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

    def call_restore(self, data: Any, from_serialized: bool = False, **kwargs) -> None:
        model_restore(self.q_ext_online, data[0], from_serialized)
        model_restore(self.q_ext_target, data[0], from_serialized)
        model_restore(self.q_int_online, data[1], from_serialized)
        model_restore(self.q_int_target, data[1], from_serialized)
        model_restore(self.emb_network, data[2], from_serialized)
        model_restore(self.lifelong_target, data[3], from_serialized)
        model_restore(self.lifelong_train, data[4], from_serialized)

    def call_backup(self, serialized: bool = False, **kwargs):
        d = [
            model_backup(self.q_ext_online, serialized),
            model_backup(self.q_int_online, serialized),
            model_backup(self.emb_network, serialized),
            model_backup(self.lifelong_target, serialized),
            model_backup(self.lifelong_train, serialized),
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


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self):
        self.q_ext_optimizer = optim.Adam(self.parameter.q_ext_online.parameters(), lr=self.config.lr_ext)
        self.q_ext_optimizer = self.config.lr_ext_scheduler.apply_torch_scheduler(self.q_ext_optimizer)
        self.q_int_optimizer = optim.Adam(self.parameter.q_int_online.parameters(), lr=self.config.lr_int)
        self.q_int_optimizer = self.config.lr_int_scheduler.apply_torch_scheduler(self.q_int_optimizer)
        self.q_criterion = nn.HuberLoss()

        self.emb_optimizer = optim.Adam(self.parameter.emb_network.parameters(), lr=self.config.episodic_lr)
        self.emb_optimizer = self.config.episodic_lr_scheduler.apply_torch_scheduler(self.emb_optimizer)
        self.emb_criterion = nn.MSELoss()

        self.lifelong_optimizer = optim.Adam(self.parameter.lifelong_train.parameters(), lr=self.config.lifelong_lr)
        self.lifelong_optimizer = self.config.lifelong_lr_scheduler.apply_torch_scheduler(self.lifelong_optimizer)
        self.lifelong_criterion = nn.MSELoss()

        self.beta_list = funcs.create_beta_list(self.config.actor_num)
        self.discount_list = funcs.create_discount_list(self.config.actor_num)

        self.sync_count = 0

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return
        batches, weights, update_args = batches

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
        ) = self.parameter.change_batches_format(batches)

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

        else:
            td_errors_int = 0.0

        # --- update memory
        if self.config.disable_int_priority:
            priorities = np.abs(td_errors_ext)
        else:
            batch_beta = np.array([self.beta_list[a] for a in actor_idx], np.float32)
            priorities = np.abs(td_errors_ext + batch_beta * td_errors_int)
        self.memory.update(update_args, priorities, self.train_count)

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

        td_errors = (target_q - q).to("cpu").detach().numpy()
        return td_errors, loss
