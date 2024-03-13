import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.rl.base import RLTrainer
from srl.rl.models.torch_ import helper
from srl.rl.models.torch_.input_block import InputBlock

from .dqn import CommonInterfaceParameter, Config


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.input_block = InputBlock(config.input_value_block, config.input_image_block, config.observation_space)
        self.hidden_block = config.hidden_block.create_block_torch(self.input_block.out_size)
        self.out_layer = nn.Linear(self.hidden_block.out_size, config.action_num)

    def forward(self, x):
        x = self.input_block(x)
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
    def create_batch_data(self, state):
        return helper.create_batch_data(state, self.config.observation_space, self.device)

    def predict_q(self, state) -> np.ndarray:
        self.q_online.eval()
        with torch.no_grad():
            q = self.q_online(state)
            q = q.to("cpu").detach().numpy()
        return q

    def predict_target_q(self, state) -> np.ndarray:
        with torch.no_grad():
            q = self.q_target(state)
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
        device = self.parameter.device
        self.parameter.q_target.to(device)
        self.parameter.q_online.to(device)

        batchs = self.memory.sample(self.batch_size)
        state, n_state, onehot_action, reward, done, next_invalid_actions = zip(*batchs)
        state = helper.stack_batch_data(state, self.config.observation_space, device)
        n_state = helper.stack_batch_data(n_state, self.config.observation_space, device)
        onehot_action = torch.tensor(np.asarray(onehot_action, dtype=np.float32)).to(device)
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done)

        target_q = self.parameter.calc_target_q(
            len(batchs),
            n_state,
            reward,
            done,
            next_invalid_actions,
        )
        target_q = torch.from_numpy(target_q).to(dtype=torch.float32).to(device)

        # --- torch train
        self.parameter.q_online.train()
        q = self.parameter.q_online(state)

        # 現在選んだアクションのQ値
        q = torch.sum(q * onehot_action, dim=1)

        loss = self.criterion(target_q, q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.lr_sch.update(self.train_count):
            lr = self.lr_sch.get_rate()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        # --- targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.load_state_dict(self.parameter.q_online.state_dict())
            self.sync_count += 1

        self.train_info = {
            "loss": loss.item(),
            "sync": self.sync_count,
            "lr": self.lr_sch.get_rate(),
        }
        self.train_count += 1
