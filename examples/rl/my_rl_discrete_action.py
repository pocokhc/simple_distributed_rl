import random
from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras  # type: ignore
import tensorflow.keras.layers as kl  # type: ignore
from srl.base.rl import DiscreteActionConfig, RLParameter, RLTrainer, RLWorker
from srl.rl.functions.model import ImageLayerType, create_input_layers
from srl.rl.registory import register


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    epsilon: float = 0.1
    test_epsilon: float = 0
    gamma: float = 0.9

    # common
    batch_size: int = 16
    memory_warmup_size: int = 100

    def __post_init__(self):
        super().__init__(self.batch_size, self.memory_warmup_size)

    @staticmethod
    def getName() -> str:
        return "MyRLDirecreteAction"

    def assert_params(self) -> None:
        super().assert_params()


register(Config, __name__)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        input_, c = create_input_layers(
            1,
            self.config.env_observation_shape,
            self.config.env_observation_type,
            ImageLayerType.DQN,
        )
        c = kl.Dense(512, activation="relu")(c)
        c = kl.Dense(self.config.nb_actions, activation="linear")(c)
        self.Q = keras.Model(input_, c)

    def restore(self, data: Optional[Any]) -> None:
        if data is None:
            return
        self.Q.set_weights(data)

    def backup(self):
        return self.Q.get_weights()


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)

        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.Huber()

    def train_on_batchs(self, batchs: list, weights: list[float]):

        # データ形式を変形
        states = []
        actions = []
        n_states = []
        rewards = []
        done_list = []
        for b in batchs:
            states.append(b["state"])
            actions.append(b["action"])
            n_states.append(b["next_state"])
            rewards.append(b["reward"])
            done_list.append(b["done"])
        states = np.asarray(states)
        n_states = np.asarray(n_states)
        done_list = np.asarray(done_list)

        # Q
        n_q = self.parameter.Q(n_states).numpy()
        target_q = rewards + (1 - done_list) * self.config.gamma * np.max(n_q, axis=1)
        with tf.GradientTape() as tape:
            q = self.parameter.Q(states)

            # 選んだアクションのQ値
            actions_onehot = tf.one_hot(actions, self.config.nb_actions)
            q = tf.reduce_sum(q * actions_onehot, axis=1)

            loss = self.loss(target_q * weights, q * weights)

        grads = tape.gradient(loss, self.parameter.Q.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.Q.trainable_variables))

        # priority
        priorities = np.abs(target_q - q) + 0.0001

        return priorities, {"loss": loss.numpy()}


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)

    def on_reset(self, state: np.ndarray, valid_actions: list[int]) -> None:
        pass

    def policy(self, state: np.ndarray, valid_actions: list[int]) -> tuple[int, Any]:
        if self.training:
            epsilon = self.config.epsilon
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
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
        valid_actions: list[int],
        next_valid_actions: list[int],
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
        priority = 0
        return batch, priority, {}

    def render(self, state: np.ndarray, valid_actions: list[int]) -> None:
        q = self.parameter.Q(state[np.newaxis, ...])[0].numpy()
        maxa = np.argmax(q)
        for a in range(self.config.nb_actions):
            if a not in valid_actions:
                continue
            if a == maxa:
                s = "*"
            else:
                s = " "
            s += f"{a:3d}: {q[a]:5.3f}"
            print(s)
