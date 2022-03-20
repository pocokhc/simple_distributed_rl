from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp
from srl.base.rl import RLTrainer, RLWorker
from srl.base.rl.config import ContinuousActionConfig
from srl.base.rl.rl import RLParameter, RLRemoteMemory
from srl.rl.functions.common_tf import compute_logprob, compute_logprob_sgp
from srl.rl.functions.model import ImageLayerType, create_input_layers_one_sequence
from srl.rl.registory import register

tfd = tfp.distributions


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(ContinuousActionConfig):

    epsilon: float = 0.1
    test_epsilon: float = 0
    gamma: float = 0.9
    lr: float = 0.001

    # std_clip_range: float = 1.0  # 勾配爆発抑制用

    @staticmethod
    def getName() -> str:
        return "ContinuousAction"


register(Config, __name__)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        # critic
        input_, c = create_input_layers_one_sequence(
            self.config.env_observation_shape,
            self.config.env_observation_type,
            ImageLayerType.DQN,
        )
        c = kl.Dense(512, activation="relu")(c)
        c = kl.Dense(1, activation="linear")(c)
        self.critic = keras.Model(input_, c)

        # actor
        input_, c = create_input_layers_one_sequence(
            self.config.env_observation_shape,
            self.config.env_observation_type,
            ImageLayerType.DQN,
        )
        c = kl.Dense(512, activation="relu")(c)
        pi_mean = kl.Dense(self.config.action_num, activation="linear")(c)
        pi_stddev = kl.Dense(self.config.action_num, activation="linear")(c)
        self.actor = keras.Model(input_, [pi_mean, pi_stddev])

    def summary(self):
        self.critic.summary()
        self.actor.summary()

    def restore(self, data: Any) -> None:
        self.critic.set_weights(data[0])
        self.actor.set_weights(data[1])

    def backup(self):
        return [
            self.critic.get_weights(),
            self.actor.get_weights(),
        ]


# ------------------------------------------------------
# RemoteMemory
# ------------------------------------------------------
class RemoteMemory(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.buffer = []

    def length(self) -> int:
        return len(self.buffer)

    def restore(self, data: Any) -> None:
        self.buffer = data

    def backup(self):
        return self.buffer

    # --------------------

    def add(self, batch: Any) -> None:
        self.buffer.append(batch)

    def sample(self):
        buffer = self.buffer
        self.buffer = []
        return buffer


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

        self.train_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):
        if self.memory.length() == 0:
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
        actions = np.asarray(actions)

        # value
        n_v = self.parameter.critic(n_states).numpy()
        advantages = []
        for i in range(len(batchs)):
            if dones[i]:
                gain = rewards[i]
            else:
                gain = rewards[i] + self.config.gamma * n_v[i][0]
            advantages.append([gain])
        advantages = np.asarray(advantages)

        # --- critic(MSE)
        with tf.GradientTape() as tape:
            v = self.parameter.critic(states)
            v_loss = tf.reduce_mean((advantages - v) ** 2)
        grads = tape.gradient(v_loss, self.parameter.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.critic.trainable_variables))

        # baseline
        advantages -= np.mean(advantages)

        # --- actor
        with tf.GradientTape() as tape:
            mean, stddev = self.parameter.actor(states)
            stddev = tf.math.exp(stddev)

            # log(π(a|s))
            new_logpi = compute_logprob_sgp(mean, stddev, actions)

            policy_loss = new_logpi * advantages
            policy_loss = -tf.reduce_mean(policy_loss)

        grads = tape.gradient(policy_loss, self.parameter.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.actor.trainable_variables))

        return {
            "v_loss": v_loss.numpy(),
            "policy_loss": policy_loss.numpy(),
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.memory = cast(RemoteMemory, self.memory)

        self.action_centor = (self.config.action_high + self.config.action_low) / 2
        self.action_scale = self.config.action_high - self.action_centor
        self.action_centor = np.asarray([self.action_centor])
        self.action_scale = np.asarray([self.action_scale])

    def on_reset(self, state: np.ndarray, valid_actions: List[int]) -> None:
        self.episode_batchs = []

    def policy(self, state: np.ndarray, valid_actions: List[int]) -> Tuple[int, Any]:

        mean, stddev = self.parameter.actor(state[np.newaxis, ...])
        stddev = tf.math.exp(stddev)

        if self.training:
            # ガウス分布に従った乱数をだす
            action = tf.random.normal(tf.shape(mean), mean=mean, stddev=stddev)
        else:
            # テストは平均
            action = mean

        # Squashed Gaussian Policy [-∞,∞] -> [1, 1]
        env_action = tf.tanh(action)

        # [-1,1] -> [low, high]
        env_action = env_action * self.action_scale + self.action_centor

        return env_action.numpy()[0], action.numpy()[0]

    def on_step(
        self,
        state: np.ndarray,
        action: Any,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        valid_actions: List[int],
        next_valid_actions: List[int],
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
        self.episode_batchs.append(batch)

        if done:
            # MC
            r = 0
            for batch in reversed(self.episode_batchs):
                r = batch["reward"] + self.config.gamma * r
                batch["reward"] = r
                self.memory.add(batch)

        return {}

    def render(self, state: np.ndarray, valid_actions: List[int], action_to_str) -> None:
        v = self.parameter.critic(state[np.newaxis, ...]).numpy()[0][0]
        mean, stddev = self.parameter.actor(state[np.newaxis, ...])

        action = tf.tanh(mean)
        action = action * self.action_scale + self.action_centor
        action = action.numpy()[0][0]
        print(f"V            : {v}")
        print(f"action mean  : {mean.numpy()[0][0]}")
        print(f"action stddev: {stddev.numpy()[0][0]}")
        print(f"action       : {action_to_str(action)}")
