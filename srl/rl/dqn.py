import random
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from srl.base.rl import DiscreteActionConfig, RLParameter, RLRemoteMemory, RLTrainer, RLWorker
from srl.rl.functions.model import ImageLayerType, create_input_layers
from srl.rl.registory import register
from tensorflow.keras import layers as kl

"""
window_length               : o (option)
Target Network              : o (+Double DQN)(option)
Huber loss function         : o
Delay update Target Network : o
Experience Replay  : o
Frame skip         : x
Annealing e-greedy : o (option)
Reward clip        : o (option)
Image preprocessor : x
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    test_epsilon: float = 0

    epsilon: float = 0.1
    # Annealing e-greedy
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.01
    exploration_steps: int = -1

    # model
    window_length: int = 1
    hidden_layer_sizes: Tuple[int, ...] = (512,)
    activation: str = "relu"
    image_layer_type: ImageLayerType = ImageLayerType.DQN

    gamma: float = 0.99  # 割引率
    lr: float = 0.001  # 学習率
    batch_size: int = 32
    capacity: int = 100_000
    memory_warmup_size: int = 1000
    target_model_update_interval: int = 100
    reward_clip: Optional[Tuple[float, float]] = None
    enable_double_dqn: bool = True

    dummy_state_val: float = 0.0

    @staticmethod
    def getName() -> str:
        return "DQN"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.window_length > 0
        assert self.memory_warmup_size < self.capacity
        assert self.batch_size < self.memory_warmup_size


register(Config, __name__)


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        in_state, c = create_input_layers(
            config.window_length,
            config.env_observation_shape,
            config.env_observation_type,
            config.image_layer_type,
        )

        # --- hidden layer
        for h in config.hidden_layer_sizes:
            c = kl.Dense(h, activation=config.activation, kernel_initializer="he_normal")(c)

        # --- out layer
        c = kl.Dense(config.nb_actions, activation="linear", kernel_initializer="truncated_normal")(c)
        self.model = keras.Model(in_state, c)

        # 重みを初期化
        dummy_state = np.zeros(shape=(1, config.window_length) + config.env_observation_shape, dtype=np.float32)
        val = self(dummy_state)
        assert val.shape == (1, config.nb_actions)

    def call(self, state):
        return self.model(state)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.q_online = _QNetwork(self.config)
        self.q_target = _QNetwork(self.config)

    def restore(self, data: Any) -> None:
        self.q_online.set_weights(data)
        self.q_target.set_weights(data)

    def backup(self) -> Any:
        return self.q_online.get_weights()

    def summary(self):
        self.q_online.model.summary()


# ------------------------------------------------------
# RemoteMemory
# ------------------------------------------------------
class RemoteMemory(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.memory = deque(maxlen=self.config.capacity)
        self.invalid_memory = deque(maxlen=self.config.capacity)

    def length(self) -> int:
        return len(self.memory)

    def restore(self, data: Any) -> None:
        self.memory = data[0]
        self.invalid_memory = data[1]

    def backup(self):
        return [self.memory, self.invalid_memory]

    # ---------------------------
    def length_invalid(self) -> int:
        return len(self.invalid_memory)

    def add(self, batch):
        self.memory.append(batch)

    def add_invalid(self, batch):
        self.invalid_memory.append(batch)

    def sample(self):
        return random.sample(self.memory, self.config.batch_size)

    def sample_invalid(self):
        return random.sample(self.invalid_memory, self.config.batch_size)

    def clear_invalid(self):
        return self.invalid_memory.clear()


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.memory = cast(RemoteMemory, self.memory)

        self.optimizer = keras.optimizers.Adam(learning_rate=self.config.lr)
        self.loss = keras.losses.Huber()

        self.train_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):

        if self.memory.length() < self.config.memory_warmup_size:
            return {}

        batchs = self.memory.sample()
        loss = self._train_on_batchs(batchs)

        # invalid action
        mem_invalid_len = self.memory.length_invalid()
        if mem_invalid_len > self.config.memory_warmup_size:
            batchs = self.memory.sample_invalid()
            self._train_on_batchs(batchs)

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())

        self.train_count += 1
        return {
            "loss": loss,
            "mem2": mem_invalid_len,
        }

    def _train_on_batchs(self, batchs):

        states = []
        actions = []
        n_states = []
        rewards = []
        dones = []
        for b in batchs:
            states.append(b["states"][:-1])
            actions.append(b["action"])
            n_states.append(b["states"][1:])
            rewards.append(b["reward"])
            dones.append(b["done"])
        states = np.asarray(states)
        n_states = np.asarray(n_states)

        # next Q
        n_q = self.parameter.q_online(n_states).numpy()
        n_q_target = self.parameter.q_target(n_states).numpy()

        # 各バッチのQ値を計算
        target_q = []
        for i in range(len(rewards)):
            reward = rewards[i]
            if dones[i]:
                gain = reward
            else:
                # DoubleDQN: indexはonlineQから選び、値はtargetQを選ぶ
                if self.config.enable_double_dqn:
                    n_act_idx = np.argmax(n_q[i])
                else:
                    n_act_idx = np.argmax(n_q_target[i])
                maxq = n_q_target[i][n_act_idx]
                gain = reward + self.config.gamma * maxq
            target_q.append(gain)
        target_q = np.asarray(target_q)

        with tf.GradientTape() as tape:
            q = self.parameter.q_online(states)

            # 現在選んだアクションのQ値
            actions_onehot = tf.one_hot(actions, self.config.nb_actions)
            q = tf.reduce_sum(q * actions_onehot, axis=1)

            loss = self.loss(target_q, q)

        grads = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.q_online.trainable_variables))

        return loss.numpy()


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.memory = cast(RemoteMemory, self.memory)

        self.dummy_state = np.full(self.config.env_observation_shape, self.config.dummy_state_val)
        self.invalid_action_reward = -1
        self.step = 0

        if self.config.exploration_steps > 0:
            self.initial_epsilon = self.config.initial_epsilon
            self.epsilon_step = (
                self.config.initial_epsilon - self.config.final_epsilon
            ) / self.config.exploration_steps
            self.final_epsilon = self.config.final_epsilon

    def on_reset(self, state: np.ndarray, valid_actions: List[int], _) -> None:
        self.recent_states = [self.dummy_state for _ in range(self.config.window_length + 1)]

        self.recent_states.pop(0)
        self.recent_states.append(state)

    def policy(self, _state: np.ndarray, valid_actions: List[int], _) -> Tuple[int, Any]:

        if self.training:
            if self.config.exploration_steps > 0:
                # Annealing ε-greedy
                epsilon = self.initial_epsilon - self.step * self.epsilon_step
                if epsilon < self.final_epsilon:
                    epsilon = self.final_epsilon
            else:
                epsilon = self.config.epsilon
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            # epsilonより低いならランダム
            action = random.choice(valid_actions)
        else:
            state = self.recent_states[1:]
            q = self.parameter.q_online(np.asarray([state]))[0].numpy()

            # valid actions以外は -inf にする
            q = np.array([(v if i in valid_actions else -np.inf) for i, v in enumerate(q)])

            # 最大値を選ぶ（複数はほぼないので無視）
            action = int(np.argmax(q))

        return action, action

    def on_step(
        self,
        state: np.ndarray,
        action: Any,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        valid_actions: List[int],
        next_valid_actions: List[int],
        _,
    ):

        self.recent_states.pop(0)
        self.recent_states.append(next_state)

        if not self.training:
            return {}
        self.step += 1

        # reward clip
        if self.config.reward_clip is not None:
            if reward < self.config.reward_clip[0]:
                reward = self.config.reward_clip[0]
            elif reward > self.config.reward_clip[1]:
                reward = self.config.reward_clip[1]

        batch = {
            "states": self.recent_states[:],
            "action": action,
            "reward": reward,
            "done": done,
        }
        self.memory.add(batch)

        # --- invalid action
        if self.invalid_action_reward > reward - 1:
            self.invalid_action_reward = reward - 1
            self.memory.clear_invalid()

        states = self.recent_states[:]
        states[-1] = self.dummy_state
        for a in range(self.config.nb_actions):
            if a in valid_actions:
                continue

            batch = {
                "states": states,
                "action": a,
                "reward": self.invalid_action_reward,
                "done": True,
            }
            self.memory.add_invalid(batch)

        return {}

    def render(self, state_: np.ndarray, valid_actions: List[int], action_to_str) -> None:
        state = self.recent_states[1:]
        q = self.parameter.q_online(np.asarray([state]))[0].numpy()
        maxa = np.argmax(q)
        for a in range(self.config.nb_actions):
            if a not in valid_actions:
                s = "x"
            else:
                s = " "
            if a == maxa:
                s += "*"
            else:
                s += " "
            s += f"{action_to_str(a)}: {q[a]:5.3f}"
            print(s)


if __name__ == "__main__":
    pass
