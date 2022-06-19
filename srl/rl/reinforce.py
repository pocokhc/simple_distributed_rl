from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kl
from srl.base.env.base import EnvRun
from srl.base.rl.algorithms.neuralnet_continuous import ContinuousActionConfig, ContinuousActionWorker
from srl.base.rl.algorithms.neuralnet_discrete import DiscreteActionConfig
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import ExperienceReplayBuffer
from srl.base.rl.remote_memory.sequence_memory import SequenceRemoteMemory
from srl.rl.functions.common_tf import compute_logprob_sgp
from srl.rl.functions.model import ImageLayerType, create_input_layers_one_sequence


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    # model
    policy_hidden_layer_sizes: Tuple[int, ...] = (64, 64, 64)
    q_hidden_layer_sizes: Tuple[int, ...] = (64, 64, 64)
    activation: str = "relu"
    image_layer_type: ImageLayerType = ImageLayerType.DQN

    gamma: float = 0.9  # 割引率
    lr: float = 0.001  # 学習率
    soft_target_update_tau: float = 0.02
    hard_target_update_interval: int = 100

    batch_size: int = 32
    capacity: int = 10_000
    memory_warmup_size: int = 1000

    def __post_init__(self):
        super().__init__()

    @staticmethod
    def getName() -> str:
        return "REINFORCE"

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
class _PolicyModel(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        in_state, c = create_input_layers_one_sequence(
            config.env_observation_shape,
            config.env_observation_type,
            config.image_layer_type,
        )

        # --- hidden layer
        for h in config.policy_hidden_layer_sizes:
            c = kl.Dense(h, activation=config.activation, kernel_initializer="he_normal")(c)
        c = kl.LayerNormalization()(c)  # 勾配爆発抑制用?

        # --- out layer
        c = kl.Dense(
            config.action_num,
            activation="softmax",
            bias_initializer="truncated_normal",
        )(c)
        self.model = keras.Model(in_state, c)

        # 重みを初期化
        dummy_state = np.zeros(shape=(1,) + config.env_observation_shape, dtype=np.float32)
        probs = self(dummy_state)
        # assert mean.shape == (1, config.action_num)
        # assert stddev.shape == (1, config.action_num)

    @tf.function
    def call(self, state):
        return self.model(state)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.policy = _PolicyModel(self.config)

    def restore(self, data: Any) -> None:
        self.q_online.set_weights(data)
        self.q_target.set_weights(data)

    def backup(self) -> Any:
        return self.q_online.get_weights()

    def summary(self):
        self.policy.model.summary()

    # ---------------------------------

    def get_probs(self, states):
        return self.policy.model(states)


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        self.train_count = 0

        self.policy_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr)

    def get_train_count(self):
        return self.train_count

    def train(self):

        if self.remote_memory.length() < self.config.memory_warmup_size:
            return {}

        loss_list = []
        for _ in range(100):
            batchs = self.remote_memory.sample(self.config.batch_size)

            states = []
            actions = []
            rewards = []
            for b in batchs:
                states.append(b["state"])
                actions.append(b["action"])
                rewards.append(b["reward"])
            states = np.asarray(states)
            actions = np.asarray(actions)
            rewards = np.asarray(rewards)

            rewards -= np.mean(rewards)

            # --- ポリシーの学習
            with tf.GradientTape() as tape:
                probs = self.parameter.get_probs(states)

                actions_onehot = tf.one_hot(actions, self.config.action_num)
                probs = tf.reduce_sum(probs * actions_onehot, axis=1)

                # logπ(a|s)
                probs = tf.clip_by_value(probs, 1e-6, 1.0)  # log(0)回避用
                logpi = tf.math.log(probs)

                loss = logpi * rewards
                policy_loss = -tf.reduce_mean(loss)  # 最大化

            grads = tape.gradient(policy_loss, self.parameter.policy.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(grads, self.parameter.policy.trainable_variables))

            loss_list.append(policy_loss.numpy())
            self.train_count += 1

        self.remote_memory.clear()
        return {"policy_loss": np.mean(loss_list)}


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(ContinuousActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

    def call_on_reset(self, state: np.ndarray) -> None:
        self.state = state
        self.history = []

    def call_policy(self, state: np.ndarray) -> Any:
        self.state = state

        probs = self.parameter.get_probs(state.reshape(1, -1))
        probs = probs.numpy()[0]

        action = np.random.choice([a for a in range(self.config.action_num)], p=probs)
        self.action = int(action)
        self.prob = probs[self.action]
        return action

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> Dict[str, Union[float, int]]:
        if not self.training:
            return {}
        self.history.append([self.state, self.action, self.prob, reward])

        if done:
            reward = 0
            for h in reversed(self.history):
                reward = h[3] + self.config.gamma * reward
                batch = {
                    "state": h[0],
                    "action": h[1],
                    "prob": h[2],
                    "reward": reward,
                    "done": done,
                }
                self.remote_memory.add(batch)

        return {}

    def render(self, env: EnvRun, player_index: int) -> None:
        # q = self.parameter.q_online(np.asarray([self.state]))[0].numpy()
        # TODO
        pass
