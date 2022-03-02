import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from srl.base.rl import DiscreteActionConfig, RLParameter, RLTrainer, RLWorker
from srl.rl.functions.model import ImageLayerType, create_input_layers
from srl.rl.registory import register
from tensorflow.keras import layers as kl

"""
DQN
    window_length(input_sequence): o
    Target Network      : o (Double DQN)
    Huber loss function : o
    Delay update Target Network: o
    Experience Replay   : o(distribute)
    Frame skip          : x
    Annealing e-greedy  : x
    Reward clip         : x
    Image preprocessor  : x
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    epsilon: float = 0.1
    test_epsilon: float = 0

    # model
    input_sequence: int = 1
    dense_units: int = 512
    image_layer_type: ImageLayerType = ImageLayerType.DQN

    gamma: float = 0.99  # 割引率
    lr: float = 0.001  # 学習率
    target_model_update_interval: int = 100

    # common
    batch_size: int = 32
    memory_warmup_size: int = 1000

    def __post_init__(self):
        super().__init__(self.batch_size, self.memory_warmup_size)

    @staticmethod
    def getName() -> str:
        return "DQN"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.input_sequence > 0


register(Config, __name__)


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        input_, c = create_input_layers(
            config.input_sequence,
            config.env_observation_shape,
            config.env_observation_type,
            config.image_layer_type,
        )

        # --- hidden layer
        c = kl.Dense(config.dense_units, activation="relu", kernel_initializer="he_normal")(c)

        # --- out layer
        c = kl.Dense(config.nb_actions, activation="linear", kernel_initializer="truncated_normal")(c)
        self.model = keras.Model(input_, c)

        # 重みを初期化
        dummy_state = np.zeros(shape=(1, config.input_sequence) + config.env_observation_shape, dtype=np.float32)
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

    def restore(self, data: Optional[Any]) -> None:
        if data is None:
            return
        self.q_online.set_weights(data)
        self.q_target.set_weights(data)

    def backup(self) -> Any:
        return self.q_online.get_weights()

    def summary(self):
        self.q_online.model.summary()

    # ---------------------------------

    def _calc_target_q(self, n_states, rewards, done_list):

        # Q値をだす
        n_q = self.q_online(n_states).numpy()
        n_q_target = self.q_target(n_states).numpy()

        # 各バッチのQ値を計算
        target_q = []
        for i in range(len(rewards)):
            reward = rewards[i]
            if done_list[i]:
                gain = reward
            else:
                # DoubleDQN: indexはQが最大を選び、値はそのtargetQを選ぶ
                n_act_idx = np.argmax(n_q[i])
                maxq = n_q_target[i][n_act_idx]
                gain = reward + self.config.gamma * maxq
            target_q.append(gain)
        target_q = np.asarray(target_q)

        return target_q


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)

        self.optimizer = keras.optimizers.Adam(learning_rate=self.config.lr)
        self.loss = keras.losses.Huber()

    def train_on_batchs(self, batchs: list, weights: List[float]):

        # データ形式を変形
        states = []
        actions = []
        n_states = []
        rewards = []
        done_list = []
        for b in batchs:
            states.append(b["states"][:-1])
            actions.append(b["action"])
            n_states.append(b["states"][1:])
            rewards.append(b["reward"])
            done_list.append(b["done"])
        states = np.asarray(states)
        n_states = np.asarray(n_states)

        # next Q
        target_q = self.parameter._calc_target_q(n_states, rewards, done_list)

        with tf.GradientTape() as tape:
            q = self.parameter.q_online(states)

            # 現在選んだアクションのQ値
            actions_onehot = tf.one_hot(actions, self.config.nb_actions)
            q = tf.reduce_sum(q * actions_onehot, axis=1)

            loss = self.loss(target_q * weights, q * weights)

        grads = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.q_online.trainable_variables))

        # priority
        priorities = np.abs(target_q - q) + 0.0001

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())

        return priorities, {"loss": loss.numpy()}


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)

    def on_reset(self, state: np.ndarray, valid_actions: List[int]) -> None:
        self.recent_states = [
            np.zeros(self.config.env_observation_shape) for _ in range(self.config.input_sequence + 1)
        ]

        self.recent_states.pop(0)
        self.recent_states.append(state)

    def policy(self, _state: np.ndarray, valid_actions: List[int]) -> Tuple[int, Any]:  # (env_action, agent_action)
        if self.training:
            epsilon = self.config.epsilon
        else:
            epsilon = self.config.test_epsilon

        # Q値を取得
        state = self.recent_states[1:]
        q = self.parameter.q_online(np.asarray([state]))[0].numpy()

        if random.random() < epsilon:
            # epsilonより低いならランダム
            action = random.choice(valid_actions)
        else:
            # valid actions以外は -inf にする
            q = np.array([(v if i in valid_actions else -np.inf) for i, v in enumerate(q)])
            # 最大値を選ぶ（複数はほぼないので無視）
            action = int(np.argmax(q))

        return action, (action, q[action])

    def on_step(
        self,
        state: np.ndarray,
        action_: Any,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        valid_actions: List[int],
        next_valid_actions: List[int],
    ):

        self.recent_states.pop(0)
        self.recent_states.append(next_state)

        if not self.training:
            return {}

        action = action_[0]
        q = action_[1]

        # priority を計算
        n_state = self.recent_states[1:]
        target_q = self.parameter._calc_target_q(np.asarray([n_state]), [reward], [done])[0]
        priority = abs(target_q - q) + 0.0001

        batch = {
            "states": self.recent_states[:],
            "action": action,
            "reward": reward,
            "done": done,
        }

        return batch, priority, {}

    def render(self, state_: np.ndarray, valid_actions: List[int]) -> None:
        state = self.recent_states[1:]
        q = self.parameter.q_online(np.asarray([state]))[0].numpy()
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


if __name__ == "__main__":
    pass
