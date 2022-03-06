from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras  # type: ignore
import tensorflow.keras.layers as kl  # type: ignore
from srl.base.rl import RLTrainer, RLWorker
from srl.base.rl.config import ContinuousActionConfig
from srl.base.rl.rl import RLParameter
from srl.rl.functions.common_tf import compute_logprob
from srl.rl.functions.model import ImageLayerType, create_input_layers_one_sequence
from srl.rl.registory import register


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(ContinuousActionConfig):

    epsilon: float = 0.1
    test_epsilon: float = 0
    gamma: float = 0.9

    std_clip_range: float = 2.0  # 勾配爆発抑制用

    # common
    batch_size: int = 16
    memory_warmup_size: int = 100

    def __post_init__(self):
        super().__init__(self.batch_size, self.memory_warmup_size)

    @staticmethod
    def getName() -> str:
        return "MyRLContinuousAction"


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

    def restore(self, data: Optional[Any]) -> None:
        if data is None:
            return
        self.critic.set_weights(data[0])
        self.actor.set_weights(data[1])

    def backup(self):
        return [
            self.critic.get_weights(),
            self.actor.get_weights(),
        ]


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)

        self.optimizer = keras.optimizers.Adam()

    def train_on_batchs(self, batchs: list, weights: List[float]):

        # データ形式を変形
        states = []
        actions = []
        old_logpi = []
        n_states = []
        rewards = []
        done_list = []
        for b in batchs:
            states.append(b["state"])
            actions.append(b["action"])
            old_logpi.append(b["logpi"])
            n_states.append(b["next_state"])
            rewards.append(b["reward"])
            done_list.append(b["done"])
        states = np.asarray(states)
        n_states = np.asarray(n_states)
        done_list = np.asarray(done_list)
        actions = np.asarray(actions)
        old_logpi = np.asarray(old_logpi)

        # value
        n_v = self.parameter.critic(n_states).numpy()
        advantages = []
        for i in range(len(batchs)):
            if done_list[i]:
                gain = rewards[i]
            else:
                gain = rewards[i] + self.config.gamma * n_v[i]
            advantages.append([gain])
        advantages = np.asarray(advantages)

        # baseline
        advantages -= np.mean(advantages)

        # --- critic(MSE)
        with tf.GradientTape() as tape:
            v = self.parameter.critic(states)
            v_loss = tf.reduce_mean((advantages - v) ** 2)
        grads = tape.gradient(v_loss, self.parameter.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.critic.trainable_variables))

        # --- actor
        with tf.GradientTape() as tape:
            mean, stddev = self.parameter.actor(states)
            stddev = tf.clip_by_value(stddev, -self.config.std_clip_range, self.config.std_clip_range)
            stddev = tf.math.exp(stddev)

            # log(π(a|s))
            new_logpi = compute_logprob(mean, stddev, actions)

            # IS
            ratio = tf.exp(new_logpi - old_logpi)

            policy_loss = new_logpi * ratio * advantages
            policy_loss = -tf.reduce_mean(policy_loss)

        grads = tape.gradient(policy_loss, self.parameter.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.actor.trainable_variables))

        # priorities
        priorities = [0 for _ in range(len(batchs))]
        return priorities, {
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

        self.action_centor = (self.config.action_high + self.config.action_low) / 2
        self.action_scale = self.config.action_high - self.action_centor
        self.action_centor = np.asarray([self.action_centor])
        self.action_scale = np.asarray([self.action_scale])

    def on_reset(self, state: np.ndarray, valid_actions: List[int]) -> None:
        pass

    def policy(self, state: np.ndarray, valid_actions: List[int]) -> Tuple[int, Any]:

        mean, stddev = self.parameter.actor(state.reshape((1, -1)))
        stddev = tf.clip_by_value(stddev, -self.config.std_clip_range, self.config.std_clip_range)
        stddev = tf.math.exp(stddev)

        if self.training:
            # ガウス分布に従った乱数をだす
            action = tf.random.normal(tf.shape(mean), mean=mean, stddev=stddev)
            logpi = compute_logprob(mean, stddev, action).numpy()[0]
        else:
            # テストは平均
            action = mean
            logpi = 0

        # [-∞,∞]([-1,1]) -> [low, high]
        env_action = action * self.action_scale + self.action_centor

        return env_action.numpy()[0], (action.numpy()[0], logpi, mean.numpy()[0])

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
        if not self.training:
            return None, 0, {}

        batch = {
            "state": state,
            "next_state": next_state,
            "action": action_[0],
            "logpi": action_[1],
            "reward": reward,
            "done": done,
        }
        priority = 0
        return (
            batch,
            priority,
            {
                "logpi": action_[1],
                "pi": np.exp(action_[1]),
                "mean": action_[2],
            },
        )

    def render(self, state: np.ndarray, valid_actions: List[int]) -> None:
        v = self.parameter.critic(state[np.newaxis, ...]).numpy()[0]
        mean = self.parameter.actor(state[np.newaxis, ...])

        action = mean * self.action_scale + self.action_centor
        action = action[0].numpy()
        print(v, mean, np.round(action))
