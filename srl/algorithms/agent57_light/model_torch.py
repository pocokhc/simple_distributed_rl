import logging
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.rl.base import RLTrainer
from srl.rl.functions import common
from srl.rl.models.torch_.dueling_network import DuelingNetworkBlock
from srl.rl.models.torch_.input_block import InputBlock

from .agent57_light import CommonInterfaceParameter, Config, RemoteMemory

logger = logging.getLogger(__name__)


class _QNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.input_ext_reward = config.input_ext_reward
        self.input_int_reward = config.input_int_reward
        self.input_action = config.input_action
        if not config.enable_intrinsic_reward:
            self.input_int_reward = False

        # --- in block
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)
        if self.in_block.use_image_layer:
            self.image_block = config.image_block_config.create_block_torch(self.in_block.out_shape)
            self.image_flatten = nn.Flatten()
            in_size = self.image_block.out_shape[0] * self.image_block.out_shape[1] * self.image_block.out_shape[2]
        else:
            flat_shape = np.zeros(config.observation_shape).flatten().shape
            in_size = flat_shape[0]

        # --- UVFA
        if self.input_ext_reward:
            in_size += 1
        if self.input_int_reward:
            in_size += 1
        if self.input_action:
            in_size += config.action_num
        in_size += config.actor_num

        # --- hidden
        self.hidden_layers = nn.ModuleList()
        if len(config.hidden_layer_sizes) > 1:
            for i in range(len(config.hidden_layer_sizes) - 1):
                self.hidden_layers.append(
                    nn.Linear(
                        in_size,
                        config.hidden_layer_sizes[i],
                    )
                )
                self.hidden_layers.append(nn.ReLU())
                in_size = config.hidden_layer_sizes[i]

        # --- out
        self.enable_dueling_network = config.enable_dueling_network
        if config.enable_dueling_network:
            self.dueling_block = DuelingNetworkBlock(
                in_size,
                config.action_num,
                config.hidden_layer_sizes[-1],
                config.dueling_network_type,
                activation="ReLU",
            )
        else:
            self.out_layers = nn.ModuleList(
                [
                    nn.Linear(in_size, config.hidden_layer_sizes[-1]),
                    nn.ReLU(),
                    nn.Linear(config.hidden_layer_sizes[-1], config.action_num),
                ]
            )

    def forward(self, inputs):
        state = inputs[0]
        reward_ext = inputs[1]
        reward_int = inputs[2]
        onehot_action = inputs[3]
        onehot_actor = inputs[4]

        # input
        x = self.in_block(state)
        if self.in_block.use_image_layer:
            x = self.image_block(x)
            x = self.image_flatten(x)

        # UVFA
        uvfa_list = [x]
        if self.input_ext_reward:
            uvfa_list.append(reward_ext)
        if self.input_int_reward:
            uvfa_list.append(reward_int)
        if self.input_action:
            uvfa_list.append(onehot_action)
        uvfa_list.append(onehot_actor)
        x = torch.cat(uvfa_list, dim=1)

        # hidden
        for layer in self.hidden_layers:
            x = layer(x)

        # out
        if self.enable_dueling_network:
            x = self.dueling_block(x)
        else:
            for layer in self.out_layers:
                x = layer(x)

        return x


# ------------------------------------------------------
# エピソード記憶部(episodic_reward)
# ------------------------------------------------------
class _EmbeddingNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        # --- in block
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)
        if self.in_block.use_image_layer:
            self.image_block = config.image_block_config.create_block_torch(self.in_block.out_shape)
            self.image_flatten = nn.Flatten()
            in_size = self.image_block.out_shape[0] * self.image_block.out_shape[1] * self.image_block.out_shape[2]
        else:
            flat_shape = np.zeros(config.observation_shape).flatten().shape
            in_size = flat_shape[0]

        # --- emb
        self.emb_block = config.episodic_emb_block.create_block_torch(in_size)

        # --- out
        self.out_block = config.episodic_out_block.create_block_torch(self.emb_block.out_shape[0] * 2)
        self.out_block_normalize = nn.LayerNorm(self.out_block.out_shape[0])
        self.out_block_out1 = nn.Linear(self.out_block.out_shape[0], config.action_num)
        self.out_block_out2 = nn.Softmax(dim=1)

    def _image_call(self, state):
        x = self.in_block(state)
        if self.in_block.use_image_layer:
            x = self.image_block(x)
            x = self.image_flatten(x)
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
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)
        if self.in_block.use_image_layer:
            self.image_block = config.image_block_config.create_block_torch(self.in_block.out_shape)
            self.image_flatten = nn.Flatten()
            in_size = self.image_block.out_shape[0] * self.image_block.out_shape[1] * self.image_block.out_shape[2]
        else:
            flat_shape = np.zeros(config.observation_shape).flatten().shape
            in_size = flat_shape[0]

        # hidden
        self.hidden_block = config.lifelong_hidden_block.create_block_torch(in_size)
        self.hidden_normalize = nn.LayerNorm(self.hidden_block.out_shape[0])

    def forward(self, x):
        x = self.in_block(x)
        if self.in_block.use_image_layer:
            x = self.image_block(x)
            x = self.image_flatten(x)
        x = self.hidden_block(x)
        x = self.hidden_normalize(x)
        return x


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(CommonInterfaceParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.device = torch.device(self.config.used_device_torch)

        self.q_ext_online = _QNetwork(self.config).to(self.device)
        self.q_ext_target = _QNetwork(self.config).to(self.device)
        self.q_int_online = _QNetwork(self.config).to(self.device)
        self.q_int_target = _QNetwork(self.config).to(self.device)
        self.emb_network = _EmbeddingNetwork(self.config).to(self.device)
        self.lifelong_target = _LifelongNetwork(self.config).to(self.device)
        self.lifelong_train = _LifelongNetwork(self.config).to(self.device)
        self.q_ext_target.eval()
        self.q_int_target.eval()
        self.lifelong_target.eval()

    def call_restore(self, data: Any, **kwargs) -> None:
        self.q_ext_online.load_state_dict(data[0])
        self.q_ext_target.load_state_dict(data[0])
        self.q_int_online.load_state_dict(data[1])
        self.q_int_target.load_state_dict(data[1])
        self.emb_network.load_state_dict(data[2])
        self.lifelong_target.load_state_dict(data[3])
        self.lifelong_train.load_state_dict(data[4])
        self.q_ext_online.to(self.device)
        self.q_ext_target.to(self.device)
        self.q_int_online.to(self.device)
        self.q_int_target.to(self.device)
        self.emb_network.to(self.device)
        self.lifelong_target.to(self.device)
        self.lifelong_train.to(self.device)

    def call_backup(self, **kwargs):
        d = [
            self.q_ext_online.to("cpu").state_dict(),
            self.q_int_online.to("cpu").state_dict(),
            self.emb_network.to("cpu").state_dict(),
            self.lifelong_target.to("cpu").state_dict(),
            self.lifelong_train.to("cpu").state_dict(),
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
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        self.q_ext_optimizer = optim.Adam(self.parameter.q_ext_online.parameters(), lr=self.config.q_ext_lr)
        self.q_int_optimizer = optim.Adam(self.parameter.q_int_online.parameters(), lr=self.config.q_int_lr)
        self.q_criterion = nn.HuberLoss()

        self.emb_optimizer = optim.Adam(self.parameter.emb_network.parameters(), lr=self.config.episodic_lr)
        self.emb_criterion = nn.MSELoss()

        self.lifelong_optimizer = optim.Adam(self.parameter.lifelong_train.parameters(), lr=self.config.lifelong_lr)
        self.lifelong_criterion = nn.MSELoss()

        self.beta_list = common.create_beta_list(self.config.actor_num)
        self.discount_list = common.create_discount_list(self.config.actor_num)

        self.train_count = 0
        self.sync_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):
        if self.remote_memory.length() < self.config.memory_warmup_size:
            return {}
        indices, batchs, weights = self.remote_memory.sample(self.config.batch_size, self.train_count)
        info = {}

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
        td_errors_ext, loss_ext = self._update_q(
            True,
            self.parameter.q_ext_online,
            self.q_ext_optimizer,
            rewards_ext,
            *_params,
        )
        info["loss_ext"] = loss_ext

        # --- intrinsic reward
        if self.config.enable_intrinsic_reward:
            self.parameter.q_int_online.to(device)
            self.parameter.q_int_target.to(device)
            td_errors_int, loss_int = self._update_q(
                False,
                self.parameter.q_int_online,
                self.q_int_optimizer,
                rewards_int,
                *_params,
            )

            # ----------------------------------------
            # embedding network
            # ----------------------------------------
            self.parameter.emb_network.train()
            actions_probs = self.parameter.emb_network([states, n_states])
            emb_loss = self.emb_criterion(actions_probs, onehot_actions)

            self.emb_optimizer.zero_grad()
            emb_loss.backward()
            self.emb_optimizer.step()

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

            info["loss_int"] = loss_int
            info["emb_loss"] = emb_loss.item()
            info["lifelong_loss"] = lifelong_loss.item()
        else:
            td_errors_int = 0.0

        # --- update memory
        if self.config.disable_int_priority:
            td_errors = td_errors_ext
        else:
            batch_beta = np.array([self.beta_list[a] for a in actor_idx], np.float32)
            td_errors = td_errors_ext + batch_beta * td_errors_int
        self.remote_memory.update(indices, batchs, td_errors)

        # --- sync target
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_ext_target.load_state_dict(self.parameter.q_ext_online.state_dict())
            self.parameter.q_int_target.load_state_dict(self.parameter.q_int_online.state_dict())
            self.sync_count += 1

        self.train_count += 1
        info["sync"] = self.sync_count
        return info

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
        return td_errors, loss.item()
