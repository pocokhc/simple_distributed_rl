import random
from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.define import EnvObservationType, RLObservationType
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.processor import Processor
from srl.base.rl.processors.image_processor import ImageProcessor
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import ExperienceReplayBuffer
from srl.rl.functions.common import create_epsilon_list, render_discrete_action
from srl.rl.models.torch.input_layer import InputLayer

"""
window_length          : -
Fixed Target Q-Network : o
Error clipping     : o
Experience Replay  : o
Frame skip         : -
Annealing e-greedy : o (config selection)
Reward clip        : o (config selection)
Image preprocessor : -

Other
    Double DQN      : o (config selection)
    invalid_actions : o
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    test_epsilon: float = 0

    epsilon: float = 0.1
    actor_epsilon: float = 0.4
    actor_alpha: float = 7.0

    # Annealing e-greedy
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.01
    exploration_steps: int = 0  # 0 : no Annealing

    # model
    hidden_layer_sizes: Tuple[int, ...] = (512,)

    discount: float = 0.99  # 割引率
    lr: float = 0.001  # 学習率
    batch_size: int = 32
    capacity: int = 100_000
    memory_warmup_size: int = 1000
    target_model_update_interval: int = 1000
    enable_reward_clip: bool = False

    # other
    enable_double_dqn: bool = True

    def set_config_by_actor(self, actor_num: int, actor_id: int) -> None:
        self.epsilon = create_epsilon_list(actor_num, epsilon=self.actor_epsilon, alpha=self.actor_alpha)[actor_id]

    # -------------------------------

    def set_processor(self) -> List[Processor]:
        return [
            ImageProcessor(
                image_type=EnvObservationType.GRAY_2ch,
                resize=(84, 84),
                enable_norm=True,
            )
        ]

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.CONTINUOUS

    @staticmethod
    def getName() -> str:
        return "DQN_torch"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.memory_warmup_size < self.capacity
        assert self.batch_size < self.memory_warmup_size
        assert len(self.hidden_layer_sizes) > 0


register(
    Config,
    __name__ + ":RemoteMemory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


# ------------------------------------------------------
# RemoteMemory
# ------------------------------------------------------
class RemoteMemory(ExperienceReplayBuffer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.init(self.config.capacity)


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.in_layer = InputLayer(config.observation_shape, config.env_observation_type)
        if self.in_layer.is_image_head():
            # (batch, 1, 84, 84)
            # -> (batch, 32, 21, 21)
            # -> (batch, 64, 11, 11)
            # -> (batch, 64, 11, 11)
            self.conv1 = nn.Conv2d(
                config.window_length, 32, kernel_size=8, stride=4, padding=3, padding_mode="replicate"
            )
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2, padding_mode="replicate")
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode="replicate")
            self.flatten = nn.Flatten()
            in_size = 11 * 11 * 64
        else:
            flat_shape = np.zeros(config.observation_shape).flatten().shape
            in_size = flat_shape[0]

        self.hidden_layers = nn.Sequential()
        self.hidden_layers.add_module("fc0", nn.Linear(in_size, config.hidden_layer_sizes[0]))
        self.hidden_layers.add_module("act0", nn.ReLU())
        for i in range(1, len(config.hidden_layer_sizes)):
            h = nn.Linear(config.hidden_layer_sizes[i - 1], config.hidden_layer_sizes[i])
            self.hidden_layers.add_module(f"fc{i}", h)
            self.hidden_layers.add_module(f"act{i}", nn.ReLU())

        self.out_layer = nn.Linear(config.hidden_layer_sizes[-1], config.action_num)

    def forward(self, x):
        x = self.in_layer(x)
        if self.in_layer.is_image_head():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flatten(x)
        x = self.hidden_layers(x)
        return self.out_layer(x)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.q_online = _QNetwork(self.config)
        self.q_target = _QNetwork(self.config)
        self.q_target.eval()

    def call_restore(self, data: Any, **kwargs) -> None:
        self.q_online.load_state_dict(data)
        self.q_target.load_state_dict(data)

    def call_backup(self, **kwargs) -> Any:
        return self.q_online.state_dict()

    def summary(self, **kwargs):
        import torchinfo

        shape = (1,) + self.config.observation_shape
        print(f"input shape={shape}")
        torchinfo.summary(self.q_online, input_size=shape)


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
        self.parameter.q_online.train()

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

        states = []
        n_states = []
        actions = []
        for b in batchs:
            states.append(b["state"])
            n_states.append(b["next_state"])
            actions.append(b["action"])
        states = torch.tensor(states)
        n_states = torch.tensor(n_states)

        # next Q
        with torch.no_grad():
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
                gain = reward + self.config.discount * maxq
            target_q[i] = gain
        target_q = torch.from_numpy(target_q.astype(np.float32))

        # --- torch train
        q = self.parameter.q_online(states)

        # 現在選んだアクションのQ値
        actions_onehot = nn.functional.one_hot(torch.tensor(actions), self.config.action_num)
        q = torch.sum(q * actions_onehot, dim=1)

        loss = self.criterion(target_q, q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        self.step_epsilon = 0

        if self.config.exploration_steps > 0:
            self.initial_epsilon = self.config.initial_epsilon
            self.epsilon_step = (
                self.config.initial_epsilon - self.config.final_epsilon
            ) / self.config.exploration_steps
            self.final_epsilon = self.config.final_epsilon

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        return {}

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        self.invalid_actions = invalid_actions
        self.state = state.astype(np.float32)

        if self.training:
            if self.config.exploration_steps > 0:
                # Annealing ε-greedy
                epsilon = self.initial_epsilon - self.step_epsilon * self.epsilon_step
                if epsilon < self.final_epsilon:
                    epsilon = self.final_epsilon
            else:
                epsilon = self.config.epsilon
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            # epsilonより低いならランダム
            action = np.random.choice([a for a in range(self.config.action_num) if a not in invalid_actions])
        else:
            self.parameter.q_online.eval()
            with torch.no_grad():
                q = self.parameter.q_online(torch.tensor(self.state[np.newaxis, ...]))
                q = q.to("cpu").detach().numpy()[0]

            # invalid actionsは -inf にする
            q = [(-np.inf if a in invalid_actions else v) for a, v in enumerate(q)]

            # 最大値を選ぶ（複数はほぼないので無視）
            action = int(np.argmax(q))

        self.action = action
        return action, {"epsilon": epsilon}

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ):
        if not self.training:
            return {}
        self.step_epsilon += 1

        # reward clip
        if self.config.enable_reward_clip:
            if reward < 0:
                reward = -1
            elif reward > 0:
                reward = 1
            else:
                reward = 0

        batch = {
            "state": self.state,
            "next_state": next_state.astype(np.float32),
            "action": self.action,
            "reward": reward,
            "done": done,
            "next_invalid_actions": next_invalid_actions,
        }
        self.remote_memory.add(batch)

        return {}

    def render_terminal(self, env, worker, **kwargs) -> None:
        self.parameter.q_online.eval()
        with torch.no_grad():
            q = self.parameter.q_online(torch.tensor(self.state[np.newaxis, ...]))
            q = q.to("cpu").detach().numpy()[0]
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        render_discrete_action(self.invalid_actions, maxa, env, _render_sub)
