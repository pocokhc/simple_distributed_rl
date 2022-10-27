import random
from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from srl.base.define import EnvObservationType, RLObservationType
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.processor import Processor
from srl.base.rl.processors.image_processor import ImageProcessor
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import ExperienceReplayBuffer
from srl.rl.functions.common import create_epsilon_list, inverse_rescaling, render_discrete_action, rescaling
from srl.rl.models.dqn_image_block import DQNImageBlock
from srl.rl.models.input_layer import create_input_layer
from srl.rl.models.mlp_block import MLPBlock


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

    discount: float = 0.99  # 割引率
    lr: float = 0.001  # 学習率
    batch_size: int = 32
    capacity: int = 100_000
    memory_warmup_size: int = 1000
    target_model_update_interval: int = 1000
    enable_reward_clip: bool = False

    # other
    enable_double_dqn: bool = True
    enable_rescale: bool = False

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

        self.in_layer = nn.Linear(config.observation_shape[0], 64)
        self.l1 = nn.Linear(64, 64)
        self.l2 = nn.Linear(64, 64)
        self.out_layer = nn.Linear(64, config.action_num)

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
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

    def call_restore(self, data: Any, **kwargs) -> None:
        self.q_online.set_weights(data)
        self.q_target.set_weights(data)

    def call_backup(self, **kwargs) -> Any:
        return self.q_online.get_weights()

    def summary(self, **kwargs):
        pass


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        # self.optimizer = keras.optimizers.Adam(learning_rate=self.config.lr)
        # self.loss = keras.losses.Huber()

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
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())
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

        # ---
        q = self.parameter.q_online(torch.tensor(states))

        # 現在選んだアクションのQ値
        actions_onehot = tf.one_hot(actions, self.config.action_num)
        q = tf.reduce_sum(q * actions_onehot, axis=1)

        # --- Compute Huber loss
        loss = F.smooth_l1_loss(target_q, q)

        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        with tf.GradientTape() as tape:
            q = self.parameter.q_online(states)

            # 現在選んだアクションのQ値
            actions_onehot = tf.one_hot(actions, self.config.action_num)
            q = tf.reduce_sum(q * actions_onehot, axis=1)

            loss = self.loss(target_q, q)

        grads = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.q_online.trainable_variables))

        return loss.numpy()


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
        self.state = state

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
            action = random.choice([a for a in range(self.config.action_num) if a not in invalid_actions])
        else:
            with torch.no_grad():
                q = self.parameter.q_online(torch.tensor(self.state))

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
            "next_state": next_state,
            "action": self.action,
            "reward": reward,
            "done": done,
            "next_invalid_actions": next_invalid_actions,
        }
        self.remote_memory.add(batch)

        return {}

    def render_terminal(self, env, worker, **kwargs) -> None:
        q = self.parameter.q_online(self.state[np.newaxis, ...])[0].numpy()
        maxa = np.argmax(q)
        if self.config.enable_rescale:
            q = inverse_rescaling(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        render_discrete_action(self.invalid_actions, maxa, env, _render_sub)
