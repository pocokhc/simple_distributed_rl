import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.rl.trainer import RLTrainer
from srl.base.spaces.multi import MultiSpace
from srl.rl.torch_.blocks.input_multi_block import InputMultiBlockConcat
from srl.rl.torch_.helper import model_backup, model_restore

from .dqn import CommonInterfaceParameter, Config, Memory


# ------------------------------------------------------
# network
# ------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        if config.observation_space.is_value():
            self.in_block = config.input_value_block.create_torch_block(config.observation_space.shape)
        elif config.observation_space.is_image():
            self.in_block = config.input_image_block.create_torch_block(config.observation_space)
        elif config.observation_space.is_multi():
            space = config.observation_space
            assert isinstance(space, MultiSpace)
            self.in_block = InputMultiBlockConcat(
                space,
                config.input_value_block,
                config.input_image_block,
                reshape_for_rnn=[False] * len(space.spaces),
            )
        else:
            raise ValueError(config.observation_space)

        self.hidden_block = config.hidden_block.create_torch_block(self.in_block.out_size)
        self.out_layer = nn.Linear(self.hidden_block.out_size, config.action_space.n)

    def forward(self, x):
        x = self.in_block(x)
        x = self.hidden_block(x)
        x = self.out_layer(x)
        return x


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(CommonInterfaceParameter):
    def setup(self):
        super().setup()
        self.device = torch.device(self.config.used_device_torch)
        self.device_str = self.config.used_device_torch
        self.torch_dtype = self.config.get_dtype("torch")

        self.q_online = QNetwork(self.config).to(self.device)
        self.q_target = QNetwork(self.config).to(self.device)
        self.q_target.eval()
        self.q_target.load_state_dict(self.q_online.state_dict())

    def call_restore(self, data: Any, from_cpu: bool = False, **kwargs) -> None:
        model_restore(self.q_online, data, from_cpu)
        model_restore(self.q_target, data, from_cpu)

    def call_backup(self, to_cpu: bool = False, **kwargs) -> Any:
        return model_backup(self.q_online, to_cpu)

    def summary(self, **kwargs):
        print(self.q_online)

    # -----------------------------------
    def pred_single_q(self, state) -> np.ndarray:
        with torch.no_grad():
            state = self.q_online.in_block.to_torch_one_batch(state, self.device, self.torch_dtype)
            q = self.q_online(state)
            q = q.detach().cpu().numpy()
        return q[0]

    def pred_batch_q(self, state) -> np.ndarray:
        with torch.no_grad():
            state = self.q_online.in_block.to_torch_batches(state, self.device, self.torch_dtype)
            q = self.q_online(state)
            q = q.detach().cpu().numpy()
        return q

    def pred_batch_target_q(self, state) -> np.ndarray:
        with torch.no_grad():
            state = self.q_target.in_block.to_torch_batches(state, self.device, self.torch_dtype)
            q = self.q_target(state)
            q = q.detach().cpu().numpy()
        return q


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.optimizer = optim.Adam(self.parameter.q_online.parameters(), lr=self.config.lr)
        self.optimizer = self.config.lr_scheduler.apply_torch_scheduler(self.optimizer)
        self.criterion = nn.HuberLoss()
        self.sync_count = 0

        self.torch_dtype = self.config.get_dtype("torch")
        self.np_dtype = self.config.get_dtype("np")
        self.device = self.parameter.device
        self.parameter.q_target.to(self.device)
        self.parameter.q_online.to(self.device)
        self.parameter.q_online.train()

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return
        batches, weights, update_args = batches

        state, n_state, onehot_action, reward, undone, next_invalid_actions = zip(*batches)
        state = self.parameter.q_online.in_block.to_torch_batches(state, self.device, self.torch_dtype)
        onehot_action = torch.tensor(np.asarray(onehot_action), dtype=self.torch_dtype, device=self.device)
        weights = torch.tensor(np.asarray(weights), dtype=self.torch_dtype, device=self.device)
        reward = np.array(reward, dtype=self.np_dtype)
        undone = np.array(undone)

        target_q = self.parameter.calc_target_q(
            len(batches),
            n_state,
            reward,
            undone,
            next_invalid_actions,
        )
        target_q = torch.tensor(target_q, dtype=self.torch_dtype, device=self.device)

        # --- torch train
        q = self.parameter.q_online(state)
        q = torch.sum(q * onehot_action, dim=1)
        loss = self.criterion(target_q * weights, q * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.info["loss"] = loss.item()

        # --- update
        priorities = np.abs((target_q - q).detach().cpu().numpy())
        self.memory.update(update_args, priorities, self.train_count)

        # --- targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.load_state_dict(self.parameter.q_online.state_dict())
            self.sync_count += 1

        self.info["sync"] = self.sync_count
        self.train_count += 1
