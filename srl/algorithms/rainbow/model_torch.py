import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.define import InfoType
from srl.base.rl.trainer import RLTrainer
from srl.rl.models.torch_ import helper
from srl.rl.models.torch_.blocks.input_block import create_in_block_out_value
from srl.rl.schedulers.scheduler import SchedulerConfig

from .rainbow import CommonInterfaceParameter, Config
from .rainbow_nomultisteps import calc_target_q


# ------------------------------------------------------
# network
# ------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.input_block = create_in_block_out_value(
            config.input_value_block,
            config.input_image_block,
            config.observation_space,
        )

        self.hidden_block = config.hidden_block.create_block_torch(
            self.input_block.out_size,
            config.action_space.n,
            enable_noisy_dense=config.enable_noisy_dense,
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.hidden_block(x)
        return x


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(CommonInterfaceParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.device = torch.device(self.config.used_device_torch)

        self.q_online = QNetwork(self.config).to(self.device)
        self.q_target = QNetwork(self.config).to(self.device)
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
        if self.config.used_device_torch != "cpu" and to_cpu:
            return copy.deepcopy(self.q_online).to("cpu").state_dict()
        else:
            return self.q_online.state_dict()

    def summary(self, **kwargs):
        print(self.q_online)

    # ----------------------------------------------
    def create_batch_data(self, state):
        return helper.create_batch_data(state, self.config.observation_space, self.device)

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
class Trainer(RLTrainer[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_sch = SchedulerConfig.create_scheduler(self.config.lr)
        self.optimizer = optim.Adam(self.parameter.q_online.parameters(), lr=self.lr_sch.get_rate())
        self.criterion = nn.HuberLoss()

        self.sync_count = 0
        self.loss = None

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        indices, batchs, weights = self.memory.sample(self.batch_size, self.train_count)

        device = self.parameter.device
        self.parameter.q_online.to(device)
        self.parameter.q_target.to(device)

        if self.config.multisteps == 1:
            target_q, states, onehot_actions = calc_target_q(self.parameter, batchs, training=True)
        else:
            target_q, states, onehot_actions = self.parameter.calc_target_q(batchs, training=True)

        states = torch.tensor(states).to(device)
        onehot_actions = torch.tensor(onehot_actions).to(device)
        weights = torch.tensor(weights).to(device)
        target_q = torch.from_numpy(target_q).to(dtype=torch.float32).to(device)

        # --- train
        self.parameter.q_online.train()
        q = self.parameter.q_online(states)
        q = torch.sum(q * onehot_actions, dim=1)

        self.loss = self.criterion(target_q * weights, q * weights)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        if self.lr_sch.update(self.train_count):
            lr = self.lr_sch.get_rate()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        # --- update
        priorities = np.abs((target_q - q).to("cpu").detach().numpy())
        self.memory.update(indices, batchs, priorities)

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.load_state_dict(self.parameter.q_online.state_dict())
            self.sync_count += 1

        self.train_count += 1

    def create_info(self) -> InfoType:
        d = {
            "sync": self.sync_count,
            "lr": self.lr_sch.get_rate(),
        }
        if self.loss is not None:
            d["loss"] = self.loss.item()
        self.loss = None
        return d
