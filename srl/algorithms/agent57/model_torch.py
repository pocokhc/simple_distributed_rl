import copy
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.rl.base import RLTrainer
from srl.rl.functions.common import create_beta_list, create_discount_list
from srl.rl.models.torch_.input_block import InputBlock

from .agent57 import CommonInterfaceParameter, Config


# ------------------------------------------------------
# network
# ------------------------------------------------------
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

        # image
        if self.in_block.use_image_layer:
            self.image_block = config.image_block.create_block_torch(self.in_block.out_shape)
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

        # --- lstm
        self.hidden_size = config.lstm_units
        self.lstm_layer = nn.LSTM(
            in_size,
            config.lstm_units,
            batch_first=True,
        )
        in_size = config.lstm_units

        # out
        self.dueling_block = config.dueling_network.create_block_torch(
            in_size,
            config.action_num,
            enable_time_distributed_layer=True,
        )

    def forward(self, inputs, hidden_states):
        state = inputs[0]
        reward_ext = inputs[1]
        reward_int = inputs[2]
        onehot_action = inputs[3]
        onehot_actor = inputs[4]

        # (batch, seq, shape) -> (batch*seq, shape)
        size = list(state.size())
        batch_size = size[0]
        seq = size[1]
        shape = size[2:]
        state = state.reshape((batch_size * seq, *shape))

        # input
        x = self.in_block(state)
        if self.in_block.use_image_layer:
            x = self.image_block(x)
            x = self.image_flatten(x)

        # (batch*seq, units) -> (batch, seq, units)
        _, units = x.size()
        x = x.view(batch_size, seq, units)

        # UVFA
        uvfa_list = [x]
        if self.input_ext_reward:
            uvfa_list.append(reward_ext)
        if self.input_int_reward:
            uvfa_list.append(reward_int)
        if self.input_action:
            uvfa_list.append(onehot_action)
        uvfa_list.append(onehot_actor)
        x = torch.cat(uvfa_list, dim=2)

        # lstm
        x, hidden_states = self.lstm_layer(x, hidden_states)

        x = self.dueling_block(x)
        return x, hidden_states

    def get_initial_state(self, batch_size, device):
        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return (h_0, c_0)


# ------------------------------------------------------
# エピソード記憶部(episodic_reward)
# ------------------------------------------------------
class _EmbeddingNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        # input
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)

        # image
        if self.in_block.use_image_layer:
            self.image_block = config.image_block.create_block_torch(
                self.in_block.out_shape,
                enable_time_distributed_layer=False,
            )
            self.image_flatten = nn.Flatten()
            in_size = self.image_block.out_shape[0] * self.image_block.out_shape[1] * self.image_block.out_shape[2]
        else:
            flat_shape = np.zeros(config.observation_shape).flatten().shape
            in_size = flat_shape[0]

        # --- emb
        self.emb_block = config.episodic_emb_block.create_block_torch(in_size)

        # --- out
        self.out_block = config.episodic_out_block.create_block_torch(self.emb_block.out_size * 2)
        self.out_block_normalize = nn.LayerNorm(self.out_block.out_size)
        self.out_block_out1 = nn.Linear(self.out_block.out_size, config.action_num)
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
            self.image_block = config.image_block.create_block_torch(
                self.in_block.out_shape,
                enable_time_distributed_layer=False,
            )
            self.image_flatten = nn.Flatten()
            in_size = self.image_block.out_shape[0] * self.image_block.out_shape[1] * self.image_block.out_shape[2]
        else:
            flat_shape = np.zeros(config.observation_shape).flatten().shape
            in_size = flat_shape[0]

        # hidden
        self.hidden_block = config.lifelong_hidden_block.create_block_torch(in_size)
        self.hidden_normalize = nn.LayerNorm(self.hidden_block.out_size)

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
        self.config: Config = self.config

        self.device = torch.device(self.config.used_device_torch)

        self.q_ext_online = _QNetwork(self.config).to(self.device)
        self.q_ext_target = _QNetwork(self.config).to(self.device)
        self.q_int_online = _QNetwork(self.config).to(self.device)
        self.q_int_target = _QNetwork(self.config).to(self.device)
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

    # ------------

    def get_initial_hidden_state_q_ext(self) -> Any:
        return self.q_ext_online.get_initial_state(1, self.device)

    def get_initial_hidden_state_q_int(self) -> Any:
        return self.q_int_online.get_initial_state(1, self.device)

    def predict_q_ext_online(self, x, hidden_state) -> Tuple[np.ndarray, Any]:
        self.q_ext_online.eval()
        with torch.no_grad():
            q, h = self.q_ext_online(
                [
                    torch.tensor(x[0]).to(self.device),
                    torch.tensor(x[1]).to(self.device),
                    torch.tensor(x[2]).to(self.device),
                    torch.tensor(x[3]).to(self.device),
                    torch.tensor(x[4]).to(self.device),
                ],
                hidden_state,
            )
            q = q.to("cpu").detach().numpy()
        return q, h

    def predict_q_int_online(self, x, hidden_state) -> Tuple[np.ndarray, Any]:
        self.q_int_online.eval()
        with torch.no_grad():
            q, h = self.q_int_online(
                [
                    torch.tensor(x[0]).to(self.device),
                    torch.tensor(x[1]).to(self.device),
                    torch.tensor(x[2]).to(self.device),
                    torch.tensor(x[3]).to(self.device),
                    torch.tensor(x[4]).to(self.device),
                ],
                hidden_state,
            )
            q = q.to("cpu").detach().numpy()
        return q, h

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

    def convert_numpy_from_hidden_state(self, h):
        return [
            h[0][0].to("cpu").detach().numpy(),
            h[1][0].to("cpu").detach().numpy(),
        ]


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_sch_ext = self.config.lr_ext.create_schedulers()
        self.lr_sch_int = self.config.lr_int.create_schedulers()
        self.lr_sch_emb = self.config.episodic_lr.create_schedulers()
        self.lr_sch_ll = self.config.lifelong_lr.create_schedulers()

        self.q_ext_optimizer = optim.Adam(self.parameter.q_ext_online.parameters(), lr=self.lr_sch_ext.get_rate())
        self.q_int_optimizer = optim.Adam(self.parameter.q_int_online.parameters(), lr=self.lr_sch_int.get_rate())
        self.q_criterion = nn.HuberLoss()

        self.emb_optimizer = optim.Adam(self.parameter.emb_network.parameters(), lr=self.lr_sch_emb.get_rate())
        self.emb_criterion = nn.MSELoss()

        self.lifelong_optimizer = optim.Adam(self.parameter.lifelong_train.parameters(), lr=self.lr_sch_ll.get_rate())
        self.lifelong_criterion = nn.MSELoss()

        self.beta_list = create_beta_list(self.config.actor_num)
        self.discount_list = create_discount_list(self.config.actor_num)

        self.sync_count = 0

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        indices, batchs, weights = self.memory.sample(self.batch_size, self.train_count)

        (
            burnin_states,
            burnin_rewards_ext,
            burnin_rewards_int,
            burnin_actions_onehot,
            burnin_actor_onehot,
            instep_states,
            instep_rewards_ext,
            instep_rewards_int,
            instep_actions_onehot,
            instep_actor_onehot,
            step_rewards_ext,
            step_rewards_int,
            step_actions_onehot,
            step_dones,
            inv_act_idx1,
            inv_act_idx2,
            inv_act_idx3,
            hidden_states_ext,
            hidden_states_int,
            discounts,
            beta_list,
            weights,
        ) = self.parameter.change_batchs_format(batchs, weights)

        device = self.parameter.device

        # hidden_states
        states_h_ext = np.stack([h[0] for h in hidden_states_ext])
        states_c_ext = np.stack([h[1] for h in hidden_states_ext])
        states_h_int = np.stack([h[0] for h in hidden_states_int])
        states_c_int = np.stack([h[1] for h in hidden_states_int])
        # (batch, seq, h) -> (seq, batch, h)
        states_h_ext = np.transpose(states_h_ext, axes=(1, 0, 2))
        states_c_ext = np.transpose(states_c_ext, axes=(1, 0, 2))
        states_h_int = np.transpose(states_h_int, axes=(1, 0, 2))
        states_c_int = np.transpose(states_c_int, axes=(1, 0, 2))
        hidden_states_ext = (torch.from_numpy(states_h_ext).to(device), torch.from_numpy(states_c_ext).to(device))
        hidden_states_int = (torch.from_numpy(states_h_int).to(device), torch.from_numpy(states_c_int).to(device))
        hidden_states_ext_t = hidden_states_ext
        hidden_states_int_t = hidden_states_int

        burnin_states = torch.from_numpy(burnin_states).to(device)
        burnin_rewards_ext = torch.from_numpy(burnin_rewards_ext).to(device)
        burnin_rewards_int = torch.from_numpy(burnin_rewards_int).to(device)
        burnin_actions_onehot = torch.from_numpy(burnin_actions_onehot).to(device)
        burnin_actor_onehot = torch.from_numpy(burnin_actor_onehot).to(device)
        instep_states = torch.from_numpy(instep_states).to(device)
        instep_rewards_ext = torch.from_numpy(instep_rewards_ext).to(device)
        instep_rewards_int = torch.from_numpy(instep_rewards_int).to(device)
        instep_actions_onehot = torch.from_numpy(instep_actions_onehot).to(device)
        instep_actor_onehot = torch.from_numpy(instep_actor_onehot).to(device)
        torch_step_actions_onehot = torch.from_numpy(step_actions_onehot).to(device)
        weights = torch.from_numpy(weights).to(device)

        _params = [
            [
                burnin_states,
                burnin_rewards_ext,
                burnin_rewards_int,
                burnin_actions_onehot,
                burnin_actor_onehot,
            ],
            [
                instep_states,
                instep_rewards_ext,
                instep_rewards_int,
                instep_actions_onehot,
                instep_actor_onehot,
            ],
            step_actions_onehot,
            torch_step_actions_onehot,
            step_dones,
            inv_act_idx1,
            inv_act_idx2,
            inv_act_idx3,
            discounts,
            weights,
            device,
        ]
        td_error_ext, _loss = self._train_q(
            self.parameter.q_ext_online,
            self.parameter.q_ext_target,
            self.q_ext_optimizer,
            self.lr_sch_ext,
            step_rewards_ext,
            hidden_states_ext,
            hidden_states_ext_t,
            *_params,
        )
        _info = {}
        _info["ext_loss"] = _loss

        if self.config.enable_intrinsic_reward:
            td_error_int, _loss = self._train_q(
                self.parameter.q_int_online,
                self.parameter.q_int_target,
                self.q_int_optimizer,
                self.lr_sch_int,
                step_rewards_int,
                hidden_states_int,
                hidden_states_int_t,
                *_params,
            )
            _info["int_loss"] = _loss

            # embedding lifelong (batch, seq_len, x) -> (batch, x)
            one_states = instep_states[:, 0, ...]
            one_n_states = instep_states[:, 1, ...]
            one_actions_onehot = instep_actions_onehot[:, 0, :]

            # ----------------------------------------
            # embedding network
            # ----------------------------------------
            self.parameter.emb_network.train()
            actions_probs = self.parameter.emb_network([one_states, one_n_states])
            emb_loss = self.emb_criterion(actions_probs, one_actions_onehot)

            self.emb_optimizer.zero_grad()
            emb_loss.backward()
            self.emb_optimizer.step()
            _info["emb_loss"] = emb_loss.item()

            if self.lr_sch_emb.update(self.train_count):
                lr = self.lr_sch_emb.get_rate()
                for param_group in self.emb_optimizer.param_groups:
                    param_group["lr"] = lr

            # ----------------------------------------
            # lifelong network
            # ----------------------------------------
            with torch.no_grad():
                lifelong_target_val = self.parameter.lifelong_target(one_states)
            self.parameter.lifelong_train.train()
            lifelong_train_val = self.parameter.lifelong_train(one_states)
            lifelong_loss = self.lifelong_criterion(lifelong_target_val, lifelong_train_val)

            self.lifelong_optimizer.zero_grad()
            lifelong_loss.backward()
            self.lifelong_optimizer.step()
            _info["lifelong_loss"] = lifelong_loss.item()

            if self.lr_sch_ll.update(self.train_count):
                lr = self.lr_sch_ll.get_rate()
                for param_group in self.lifelong_optimizer.param_groups:
                    param_group["lr"] = lr

        else:
            td_error_int = 0

        if self.config.disable_int_priority:
            priorities = np.abs(td_error_ext)
        else:
            priorities = np.abs(td_error_ext + beta_list * td_error_int)

        self.memory.update((indices, batchs, priorities))

        # --- sync target
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_ext_target.load_state_dict(self.parameter.q_ext_online.state_dict())
            self.parameter.q_int_target.load_state_dict(self.parameter.q_int_online.state_dict())
            self.sync_count += 1

        _info["sync"] = self.sync_count
        self.train_count += 1
        self.train_info = _info

    def _train_q(
        self,
        model_q_online,
        model_q_target,
        optimizer,
        lr_sch,
        step_rewards,
        hidden_states,
        hidden_states_t,
        #
        in_burnin,
        in_steps,
        step_actions_onehot,
        torch_step_actions_onehot,
        step_dones,
        inv_act_idx1,
        inv_act_idx2,
        inv_act_idx3,
        discounts,
        weights,
        device,
    ):
        with torch.no_grad():
            # burnin
            if self.config.burnin > 0:
                _, hidden_states = model_q_online(in_burnin, hidden_states)
                _, hidden_states_t = model_q_target(in_burnin, hidden_states_t)

            # targetQ
            q_target, _ = model_q_target(in_steps, hidden_states_t)
            q_target = q_target.to("cpu").detach().numpy()

        model_q_online.train()
        q, _ = model_q_online(in_steps, hidden_states)
        action_q = torch.sum(q[:, :-1, :] * torch_step_actions_onehot, dim=2)
        np_action_q = action_q.to("cpu").clone().detach().numpy()

        np_target_q = self.parameter.calc_target_q(
            q.to("cpu").clone().detach().numpy(),
            q_target,
            np_action_q,
            step_rewards,
            step_actions_onehot,
            step_dones,
            inv_act_idx1,
            inv_act_idx2,
            inv_act_idx3,
            discounts,
        )
        target_q = torch.from_numpy(np_target_q).to(device)

        # (batch, seq, shape) ->  (seq, batch, shape)
        action_q = torch.transpose(action_q, 1, 0)
        np_action_q = np.transpose(np_action_q, (1, 0))

        # --- update Q
        loss = self.q_criterion(target_q * weights, action_q * weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # lr_schedule
        if lr_sch.update(self.train_count):
            lr = lr_sch.get_rate()
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        td_errors = np.mean(np_action_q - np_target_q, axis=0)
        return td_errors, loss.item()
