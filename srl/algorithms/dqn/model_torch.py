from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.rl.base import RLTrainer
from srl.rl.functions.common import inverse_rescaling, rescaling
from srl.rl.models.torch_.input_block import InputBlock
from srl.utils import common

from .dqn import CommonInterfaceParameter, Config, RemoteMemory, Worker


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)
        if self.in_block.use_image_layer:
            self.image_block = config.image_block_config.create_block_torch(self.in_block.out_shape)
            self.flatten = nn.Flatten()
            in_size = self.image_block.out_shape[0] * self.image_block.out_shape[1] * self.image_block.out_shape[2]
        else:
            flat_shape = np.zeros(config.observation_shape).flatten().shape
            in_size = flat_shape[0]

        self.hidden_block = config.hidden_block_config.create_block_torch(in_size)
        self.out_layer = nn.Linear(self.hidden_block.out_shape[0], config.action_num)

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
        self.config = cast(Config, self.config)

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
        if common.is_package_installed("torchinfo"):
            import torchinfo

            shape = (1,) + self.config.observation_shape
            print(f"input shape={shape}")
            torchinfo.summary(self.q_online, input_size=shape)
        else:
            print(self.q_online)

    # -----------------------------------

    def get_q(self, state: np.ndarray, worker: Worker):
        self.q_online.eval()
        with torch.no_grad():
            q = self.q_online(torch.tensor(state).to(self.device))
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

        self.optimizer = optim.Adam(self.parameter.q_online.parameters(), lr=self.config.lr)
        self.criterion = nn.HuberLoss()

        self.train_count = 0
        self.sync_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):
        if self.remote_memory.length() < self.config.memory_warmup_size:
            return {}

        batchs = self.remote_memory.sample(self.config.batch_size)
        loss = self._train_on_batchs(batchs)

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.load_state_dict(self.parameter.q_online.state_dict())
            self.sync_count += 1

        self.train_count += 1
        return {"loss": loss, "sync": self.sync_count}

    def _train_on_batchs(self, batchs):
        device = self.parameter.device
        self.parameter.q_online.to(device)
        self.parameter.q_target.to(device)

        states = []
        n_states = []
        actions = []
        for b in batchs:
            states.append(b["state"])
            n_states.append(b["next_state"])
            actions.append(b["action"])
        states = torch.tensor(np.asarray(states).astype(np.float32)).to(device)
        n_states = torch.tensor(np.asarray(n_states).astype(np.float32)).to(device)
        actions_onehot = np.identity(self.config.action_num)[actions].astype(np.float32)
        actions_onehot = torch.tensor(actions_onehot).to(device)

        # next Q
        with torch.no_grad():
            self.parameter.q_online.eval()
            n_q = self.parameter.q_online(n_states)
            n_q_target = self.parameter.q_target(n_states)
            n_q = n_q.to("cpu").detach().numpy()
            n_q_target = n_q_target.to("cpu").detach().numpy()

        # 各バッチのQ値を計算
        target_q = np.zeros(len(batchs))
        for i, b in enumerate(batchs):
            reward = b["reward"]
            done = b["done"]
            next_invalid_actions = b["next_invalid_actions"]
            if done:
                gain = reward
            else:
                # DoubleDQN: indexはonlineQから選び、値はtargetQを選ぶ
                if self.config.enable_double_dqn:
                    n_q[i] = [(-np.inf if a in next_invalid_actions else v) for a, v in enumerate(n_q[i])]
                    n_act_idx = np.argmax(n_q[i])
                else:
                    n_q_target[i] = [
                        (-np.inf if a in next_invalid_actions else v) for a, v in enumerate(n_q_target[i])
                    ]
                    n_act_idx = np.argmax(n_q_target[i])
                maxq = n_q_target[i][n_act_idx]
                if self.config.enable_rescale:
                    maxq = inverse_rescaling(maxq)
                gain = reward + self.config.discount * maxq
            if self.config.enable_rescale:
                gain = rescaling(gain)
            target_q[i] = gain
        target_q = torch.from_numpy(target_q.astype(np.float32)).to(device)

        # --- torch train
        self.parameter.q_online.train()
        q = self.parameter.q_online(states)

        # 現在選んだアクションのQ値
        q = torch.sum(q * actions_onehot, dim=1)

        loss = self.criterion(target_q, q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
