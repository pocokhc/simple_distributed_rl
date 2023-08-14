from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.rl.base import RLTrainer
from srl.rl.models.torch_.input_block import InputBlock

from .dqn import CommonInterfaceParameter, Config, RemoteMemory


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)
        if self.in_block.use_image_layer:
            self.image_block = config.image_block.create_block_torch(
                self.in_block.out_shape,
                enable_time_distributed_layer=False,
            )
            self.flatten = nn.Flatten()
            in_size = self.image_block.out_shape[0] * self.image_block.out_shape[1] * self.image_block.out_shape[2]
        else:
            flat_shape = np.zeros(config.observation_shape).flatten().shape
            in_size = flat_shape[0]

        self.hidden_block = config.hidden_block.create_block_torch(in_size)
        self.out_layer = nn.Linear(self.hidden_block.out_size, config.action_num)

    def forward(self, x):
        x = self.in_block(x)
        if self.in_block.use_image_layer:
            x = self.image_block(x)
            x = self.flatten(x)
        x = self.hidden_block(x)
        return self.out_layer(x)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(CommonInterfaceParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.device = torch.device(self.config.used_device_torch)

        self.q_online = _QNetwork(self.config).to(self.device)
        self.q_target = _QNetwork(self.config).to(self.device)
        self.q_target.eval()

    def call_restore(self, data: Any, **kwargs) -> None:
        self.q_online.load_state_dict(data)
        self.q_target.load_state_dict(data)
        self.q_online.to(self.device)
        self.q_target.to(self.device)

    def call_backup(self, **kwargs) -> Any:
        return self.q_online.to("cpu").state_dict()

    def summary(self, **kwargs):
        print(self.q_online)

    # -----------------------------------

    def predict_q(self, state: np.ndarray) -> np.ndarray:
        self.q_online.eval()
        with torch.no_grad():
            q = self.q_online(torch.tensor(state).to(self.device))
            q = q.to("cpu").detach().numpy()
        return q

    def predict_target_q(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            q = self.q_target(torch.tensor(state).to(self.device))
            q = q.to("cpu").detach().numpy()
        return q


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter
        self.remote_memory: RemoteMemory = self.remote_memory

        self.lr_sch = self.config.lr.create_schedulers()

        self.optimizer = optim.Adam(self.parameter.q_online.parameters(), lr=self.lr_sch.get_rate(0))
        self.criterion = nn.HuberLoss()

        self.train_count = 0
        self.sync_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):
        if self.remote_memory.length() < self.config.memory_warmup_size:
            return {}

        device = self.parameter.device
        self.parameter.q_online.to(device)
        self.parameter.q_target.to(device)

        indices, batchs, weights = self.remote_memory.sample(self.config.batch_size, self.train_count)
        target_q, states, onehot_actions = self.parameter.calc_target_q(batchs, training=True)

        states = torch.tensor(states).to(device)
        onehot_actions = torch.tensor(onehot_actions).to(device)
        weights = torch.tensor(weights).to(device)
        target_q = torch.from_numpy(target_q).to(dtype=torch.float32).to(device)

        # --- torch train
        self.parameter.q_online.train()
        q = self.parameter.q_online(states)

        # 現在選んだアクションのQ値
        q = torch.sum(q * onehot_actions, dim=1)

        loss = self.criterion(target_q * weights, q * weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        lr = self.lr_sch.get_rate(self.train_count)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # --- update
        td_errors = (target_q - q).to("cpu").detach().numpy()
        self.remote_memory.update(indices, batchs, td_errors)

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.load_state_dict(self.parameter.q_online.state_dict())
            self.sync_count += 1

        self.train_count += 1
        return {
            "loss": loss.item(),
            "sync": self.sync_count,
            "lr": lr,
        }
