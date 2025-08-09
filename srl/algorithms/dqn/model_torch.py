from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.rl.trainer import RLTrainer
from srl.rl.torch_.helper import model_backup, model_restore

from .dqn import CommonInterfaceParameter, Config, Memory


# ------------------------------------------------------
# network
# ------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.in_block = config.input_block.create_torch_block(config)
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
        self.np_dtype = self.config.get_dtype("np")

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

    # -----------------------------------
    def pred_q(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state2 = torch.tensor(np.asarray(state, dtype=self.np_dtype), device=self.device)
            q = self.q_online(state2)
            q = q.detach().cpu().numpy()
        return q

    def pred_target_q(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state2 = torch.tensor(np.asarray(state, dtype=self.np_dtype), device=self.device)
            q = self.q_target(state2)
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
        state = torch.tensor(np.asarray(state, dtype=self.np_dtype), device=self.device)
        onehot_action = torch.tensor(np.asarray(onehot_action, dtype=self.np_dtype), device=self.device)
        weights = torch.tensor(np.asarray(weights, dtype=self.np_dtype), device=self.device)
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
