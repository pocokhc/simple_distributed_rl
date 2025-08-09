from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.rl.trainer import RLTrainer
from srl.rl.torch_.helper import model_backup, model_restore

from .rainbow import CommonInterfaceParameter, Config, Memory
from .rainbow_nomultisteps import calc_target_q


class QNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.in_block = config.input_block.create_torch_block(config)
        self.hidden_block = config.hidden_block.create_torch_block(
            self.in_block.out_size,
            config.action_space.n,
            enable_noisy_dense=config.enable_noisy_dense,
        )

    def forward(self, x):
        x = self.in_block(x)
        x = self.hidden_block(x)
        return x


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

    def call_restore(self, data: Any, from_serialized: bool = False, **kwargs) -> None:
        model_restore(self.q_online, data, from_serialized)
        model_restore(self.q_target, data, from_serialized)

    def call_backup(self, serialized: bool = False, **kwargs) -> Any:
        return model_backup(self.q_online, serialized)

    def summary(self, **kwargs):
        print(self.q_online)

    # ----------------------------------------------
    def pred_q(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state2 = torch.tensor(np.asarray(state, dtype=self.np_dtype), device=self.device)
            q = self.q_online(state2)
            q = q.detach().cpu().numpy()
        return q

    def pred_target_q(self, state) -> np.ndarray:
        with torch.no_grad():
            state2 = torch.tensor(np.asarray(state, dtype=self.np_dtype), device=self.device)
            q = self.q_target(state2)
            q = q.detach().cpu().numpy()
        return q


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.optimizer = optim.Adam(self.parameter.q_online.parameters(), lr=self.config.lr)
        self.optimizer = self.config.lr_scheduler.apply_torch_scheduler(self.optimizer)
        self.criterion = nn.HuberLoss()
        self.sync_count = 0

        self.torch_dtype = self.config.get_dtype("torch")
        self.np_dtype = self.config.get_dtype("np")
        self.device = self.parameter.device
        self.device_str = self.config.used_device_torch
        self.parameter.q_online.to(self.device)
        self.parameter.q_target.to(self.device)
        self.parameter.q_online.train()

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return
        batches, weights, update_args = batches

        if self.config.multisteps == 1:
            target_q, states, onehot_actions = calc_target_q(self.parameter, batches, np_dtype=self.np_dtype)
        else:
            target_q, states, onehot_actions = self.parameter.calc_target_q(batches)

        states = torch.tensor(np.asarray(states), dtype=self.torch_dtype, device=self.device)
        onehot_actions = torch.tensor(np.asarray(onehot_actions), dtype=self.torch_dtype, device=self.device)
        weights = torch.tensor(np.asarray(weights), dtype=self.torch_dtype, device=self.device)
        target_q = torch.tensor(np.asarray(target_q), dtype=self.torch_dtype, device=self.device)

        # --- train
        self.parameter.q_online.train()
        q = self.parameter.q_online(states)
        q = torch.sum(q * onehot_actions, dim=1)
        loss = self.criterion(target_q * weights, q * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.info["loss"] = loss.item()

        # --- update
        priorities = np.abs((target_q - q).detach().cpu().numpy())
        self.memory.update(update_args, priorities, self.train_count)

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.load_state_dict(self.parameter.q_online.state_dict())
            self.sync_count += 1

        self.info["sync"] = self.sync_count
        self.train_count += 1
