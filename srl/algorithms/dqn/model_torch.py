import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.define import InfoType
from srl.base.rl.trainer import RLTrainer
from srl.rl.models.torch_.blocks.input_block import create_in_block_out_value
from srl.rl.schedulers.scheduler import SchedulerConfig

from .dqn import CommonInterfaceParameter, Config


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
        # backupでdeviceを変えるとtrainerとの並列処理でバグの可能性あり
        if self.config.used_device_torch != "cpu" and to_cpu:
            return copy.deepcopy(self.q_online).to("cpu").state_dict()
        else:
            return self.q_online.state_dict()

    def summary(self, **kwargs):
        print(self.q_online)

    # -----------------------------------
    def pred_single_q(self, state) -> np.ndarray:
        state = self.q_online.input_block.create_batch_single_data(state, self.device)
        with torch.no_grad():
            q = self.q_online(state)
            q = q.to("cpu").detach().numpy()
        return q[0]

    def pred_batch_q(self, state) -> np.ndarray:
        state = self.q_online.input_block.create_batch_stack_data(state, self.device)
        with torch.no_grad():
            q = self.q_online(state)
            q = q.to("cpu").detach().numpy()
        return q

    def pred_batch_target_q(self, state) -> np.ndarray:
        state = self.q_online.input_block.create_batch_stack_data(state, self.device)
        with torch.no_grad():
            q = self.q_target(state)
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
        device = self.parameter.device
        self.parameter.q_target.to(device)
        self.parameter.q_online.to(device)

        batchs = self.memory.sample(self.batch_size)
        state, n_state, onehot_action, reward, done, next_invalid_actions = zip(*batchs)
        state = self.parameter.q_online.input_block.create_batch_stack_data(state, device)
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
        q = self.parameter.q_online(state)

        # 現在選んだアクションのQ値
        q = torch.sum(q * onehot_action, dim=1)

        self.loss = self.criterion(target_q, q)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        if self.lr_sch.update(self.train_count):
            lr = self.lr_sch.get_rate()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        # --- targetと同期
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
