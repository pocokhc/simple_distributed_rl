import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.rl.base import RLTrainer
from srl.rl.models.torch_.input_block import InputBlock

from .dqn import CommonInterfaceParameter, Config


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
        self.q_target.load_state_dict(self.q_online.state_dict())

    def call_restore(self, data: Any, from_cpu: bool = False, **kwargs) -> None:
        if self.config.used_device_torch != "cpu" and from_cpu:
            self.q_online.to("cpu").load_state_dict(data)
            self.q_target.to("cpu").load_state_dict(data)
            self.q_online.to(self.device)
            self.q_target.to(self.device)
        else:
            self.q_online.load_state_dict(data)
            self.q_target.load_state_dict(data)

    def call_backup(self, to_cpu: bool = False, **kwargs) -> Any:
        # backupでdeviceを変えるとtrainerとの並列処理でバグの可能性あり
        if self.config.used_device_torch != "cpu" and to_cpu:
            return copy.deepcopy(self.q_online).to("cpu").state_dict()
        else:
            return self.q_online.state_dict()

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

        self.lr_sch = self.config.lr.create_schedulers()

        self.optimizer = optim.Adam(self.parameter.q_online.parameters(), lr=self.lr_sch.get_rate())
        self.criterion = nn.HuberLoss()

        self.sync_count = 0

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        batchs = self.memory.sample(self.batch_size, self.train_count)

        device = self.parameter.device
        self.parameter.q_online.to(device)
        self.parameter.q_target.to(device)

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

        if self.lr_sch.update(self.train_count):
            lr = self.lr_sch.get_rate()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        # --- update
        priorities = np.abs((target_q - q).to("cpu").detach().numpy())
        self.memory_update((indices, batchs, priorities))

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.load_state_dict(self.parameter.q_online.state_dict())
            self.sync_count += 1

        self.train_info = {
            "loss": loss.item(),
            "sync": self.sync_count,
            "lr": self.lr_sch.get_rate(),
        }
        self.train_count += 1
