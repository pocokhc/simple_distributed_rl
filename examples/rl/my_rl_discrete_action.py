import random
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kl
from srl.base.rl import DiscreteActionConfig, RLParameter, RLTrainer, RLWorker
from srl.base.rl.rl import RLRemoteMemory
from srl.rl.functions.model import ImageLayerType, create_input_layers_one_sequence
from srl.rl.registory import register


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    epsilon: float = 0.1
    test_epsilon: float = 0
    gamma: float = 0.9
    batch_size: int = 16
    memory_warmup_size: int = 100
    capacity: int = 100_000

    @staticmethod
    def getName() -> str:
        return "MyRLDirecreteAction"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.memory_warmup_size < self.capacity
        assert self.batch_size < self.memory_warmup_size


register(Config, __name__)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        in_state, c = create_input_layers_one_sequence(
            self.config.env_observation_shape,
            self.config.env_observation_type,
            ImageLayerType.DQN,
        )
        c = kl.Dense(512, activation="relu")(c)
        c = kl.Dense(self.config.nb_actions, activation="linear")(c)
        self.Q = keras.Model(in_state, c)

    def restore(self, data: Any) -> None:
        self.Q.set_weights(data)

    def backup(self):
        return self.Q.get_weights()

    def summary(self):
        self.Q.summary()


# ------------------------------------------------------
# RemoteMemory
# ------------------------------------------------------
class RemoteMemory(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.memory = deque(maxlen=self.config.capacity)

    def length(self) -> int:
        return len(self.memory)

    def restore(self, data: Any) -> None:
        self.memory = data

    def backup(self):
        return self.memory.copy()

    # ---------------------------
    def add(self, batch):
        self.memory.append(batch)

    def sample(self):
        return random.sample(self.memory, self.config.batch_size)


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.memory = cast(RemoteMemory, self.memory)

        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.Huber()

        self.train_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):
        if self.memory.length() < self.config.memory_warmup_size:
            return {}

        batchs = self.memory.sample()

        states = []
        actions = []
        n_states = []
        rewards = []
        dones = []
        for b in batchs:
            states.append(b["state"])
            actions.append(b["action"])
            n_states.append(b["next_state"])
            rewards.append(b["reward"])
            dones.append(b["done"])
        states = np.asarray(states)
        n_states = np.asarray(n_states)
        dones = np.asarray(dones)

        # Q
        n_q = self.parameter.Q(n_states).numpy()
        target_q = rewards + (1 - dones) * self.config.gamma * np.max(n_q, axis=1)
        with tf.GradientTape() as tape:
            q = self.parameter.Q(states)

            # 選んだアクションのQ値
            actions_onehot = tf.one_hot(actions, self.config.nb_actions)
            q = tf.reduce_sum(q * actions_onehot, axis=1)

            loss = self.loss(target_q, q)

        grads = tape.gradient(loss, self.parameter.Q.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.Q.trainable_variables))

        return {"loss": loss.numpy()}


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.memory = cast(RemoteMemory, self.memory)

    def on_reset(self, state: np.ndarray, valid_actions: List[int], _) -> None:
        if self.training:
            self.epsilon = self.config.epsilon
        else:
            self.epsilon = self.config.test_epsilon

    def policy(self, state: np.ndarray, valid_actions: List[int], _) -> Tuple[int, Any]:

        if random.random() < self.epsilon:
            # epsilonより低いならランダム
            action = random.choice(valid_actions)
        else:
            q = self.parameter.Q(np.asarray([state]))[0].numpy()
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
        if not self.training:
            return {}

        batch = {
            "state": state,
            "next_state": next_state,
            "action": action,
            "reward": reward,
            "done": done,
        }
        self.memory.add(batch)
        return {}

    def render(self, state: np.ndarray, valid_actions: List[int], action_to_str) -> None:
        q = self.parameter.Q(state[np.newaxis, ...])[0].numpy()
        maxa = np.argmax(q)
        for a in range(self.config.nb_actions):
            if a not in valid_actions:
                continue
            if a == maxa:
                s = "*"
            else:
                s = " "
            s += f"{action_to_str(a)}: {q[a]:5.3f}"
            print(s)
