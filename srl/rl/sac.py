from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import tensorflow as tf
from srl.base.rl.algorithms.continuous_action import ContinuousActionConfig, ContinuousActionWorker
import tensorflow.keras as keras
import tensorflow.keras.layers as kl
from srl.base.define import RLObservationType
from srl.base.env.base import EnvRun
from srl.base.rl.algorithms.continuous_action import ContinuousActionConfig, ContinuousActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import ExperienceReplayBuffer
from srl.rl.functions.common_tf import compute_logprob_sgp
from srl.rl.functions.model import ImageLayerType, create_input_layers_one_sequence

"""
DDPG
    Replay buffer       : o
    Target Network(soft): o
    Target Network(hard): o
    Add action noise    : x
TD3
    Clipped Double Q learning : o
    Target Policy Smoothing   : x
    Delayed Policy Update     : x
SAC
    Squashed Gaussian Policy: o
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(ContinuousActionConfig):

    # model
    policy_hidden_layer_sizes: Tuple[int, ...] = (64, 64, 64)
    q_hidden_layer_sizes: Tuple[int, ...] = (64, 64, 64)
    activation: str = "relu"
    image_layer_type: ImageLayerType = ImageLayerType.DQN

    gamma: float = 0.9  # 割引率
    lr: float = 0.005  # 学習率
    soft_target_update_tau: float = 0.02
    hard_target_update_interval: int = 100

    batch_size: int = 32
    capacity: int = 10_000
    memory_warmup_size: int = 500

    def __post_init__(self):
        super().__init__()

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.CONTINUOUS

    @staticmethod
    def getName() -> str:
        return "SAC"

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
            config.observation_shape,
            config.env_observation_type,
            config.image_layer_type,
        )

        # --- hidden layer
        for h in config.policy_hidden_layer_sizes:
            c = kl.Dense(h, activation=config.activation, kernel_initializer="he_normal")(c)
        c = kl.LayerNormalization()(c)  # 勾配爆発抑制用?

        # --- out layer
        pi_mean = kl.Dense(
            config.action_num,
            activation="linear",
            kernel_initializer="truncated_normal",
            bias_initializer="truncated_normal",
        )(c)
        pi_stddev = kl.Dense(
            config.action_num,
            activation="linear",
            kernel_initializer="truncated_normal",
            bias_initializer="truncated_normal",
        )(c)
        self.model = keras.Model(in_state, [pi_mean, pi_stddev])

        # 重みを初期化
        dummy_state = np.zeros(shape=(1,) + config.observation_shape, dtype=np.float32)
        action, mean, stddev, action_org = self(dummy_state)
        assert mean.shape == (1, config.action_num)
        assert stddev.shape == (1, config.action_num)

    @tf.function
    def call(self, state):
        mean, stddev = self.model(state)

        # σ > 0
        stddev = tf.exp(stddev)

        # Reparameterization trick
        normal_random = tf.random.normal(mean.shape, mean=0.0, stddev=1.0)
        action_org = mean + stddev * normal_random

        # Squashed Gaussian Policy
        action = tf.tanh(action_org)

        return action, mean, stddev, action_org


class _DualQNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        in_state, c = create_input_layers_one_sequence(
            config.observation_shape,
            config.env_observation_type,
            config.image_layer_type,
        )
        in_action = kl.Input(shape=(config.action_num,))
        c = kl.Concatenate()([c, in_action])

        for h in config.q_hidden_layer_sizes:
            c = kl.Dense(h, activation=config.activation, kernel_initializer="he_normal")(c)
        c = kl.LayerNormalization()(c)  # 勾配爆発抑制用?

        # q1
        q1 = kl.Dense(
            1, activation="linear", kernel_initializer="truncated_normal", bias_initializer="truncated_normal"
        )(c)

        # q2
        q2 = kl.Dense(
            1, activation="linear", kernel_initializer="truncated_normal", bias_initializer="truncated_normal"
        )(c)

        # out layer
        self.model = keras.Model([in_state, in_action], [q1, q2])

        # 重みを初期化
        dummy_state = np.zeros(shape=(1,) + config.observation_shape, dtype=np.float32)
        dummy_action = np.zeros(shape=(1, config.action_num), dtype=np.float32)
        _q1, _q2 = self(dummy_state, dummy_action)
        assert _q1.shape == (1, 1)
        assert _q2.shape == (1, 1)

    def call(self, state, action):
        return self.model([state, action])


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.policy = _PolicyModel(self.config)
        self.q_online = _DualQNetwork(self.config)
        self.q_target = _DualQNetwork(self.config)

    def restore(self, data: Any) -> None:
        self.policy.set_weights(data[0])
        self.q_online.set_weights(data[1])
        self.q_target.set_weights(data[1])

    def backup(self) -> Any:
        return [
            self.policy.get_weights(),
            self.q_online.get_weights(),
        ]

    def summary(self):
        self.policy.model.summary()
        self.q_online.model.summary()


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

        self.q_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr)
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr)
        self.alpha_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr)

        # エントロピーαの目標値、-1×アクション数が良いらしい
        self.target_entropy = -1 * self.config.action_num

        # エントロピーα自動調整用
        self.log_alpha = tf.Variable(0.5, dtype=tf.float32)

    def get_train_count(self):
        return self.train_count

    def train(self):
        if self.remote_memory.length() < self.config.memory_warmup_size:
            return {}
        batchs = self.remote_memory.sample(self.config.batch_size)

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
        actions = np.asarray(actions)
        dones = np.asarray(dones).reshape((-1, 1))
        rewards = np.asarray(rewards).reshape((-1, 1))

        # 方策エントロピーの反映率αを計算
        alpha = tf.math.exp(self.log_alpha)

        # ポリシーより次の状態のアクションを取得
        n_actions, n_means, n_stddevs, n_action_orgs = self.parameter.policy(n_states)
        # 次の状態のアクションのlogpiを取得(Squashed Gaussian Policy時)
        n_logpi = compute_logprob_sgp(n_means, n_stddevs, n_action_orgs)

        # 2つのQ値から小さいほうを採用(Clipped Double Q learning)して、
        # Q値を計算 : reward if done else (reward + gamma * n_qval) - (alpha * H)
        n_q1, n_q2 = self.parameter.q_target(n_states, n_actions)
        q_vals = rewards + (1 - dones) * self.config.gamma * tf.minimum(n_q1, n_q2) - (alpha * n_logpi)

        # --- Qモデルの学習
        with tf.GradientTape() as tape:
            q1, q2 = self.parameter.q_online(states, actions)
            loss1 = tf.reduce_mean(tf.square(q_vals - q1))
            loss2 = tf.reduce_mean(tf.square(q_vals - q2))
            q_loss = (loss1 + loss2) / 2

        grads = tape.gradient(q_loss, self.parameter.q_online.trainable_variables)
        self.q_optimizer.apply_gradients(zip(grads, self.parameter.q_online.trainable_variables))

        # --- ポリシーの学習
        with tf.GradientTape() as tape:
            # アクションを出力
            selected_actions, means, stddevs, action_orgs = self.parameter.policy(states)

            # logπ(a|s) (Squashed Gaussian Policy)
            logpi = compute_logprob_sgp(means, stddevs, action_orgs)

            # Q値を出力、小さいほうを使う
            q1, q2 = self.parameter.q_online(states, selected_actions)
            q_min = tf.minimum(q1, q2)

            # alphaは定数扱いなので勾配が流れないようにする
            policy_loss = q_min - (tf.stop_gradient(alpha) * logpi)

            policy_loss = -tf.reduce_mean(policy_loss)  # 最大化

        grads = tape.gradient(policy_loss, self.parameter.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(grads, self.parameter.policy.trainable_variables))

        # --- 方策エントロピーαの自動調整
        _, means, stddevs, action_orgs = self.parameter.policy(states)
        logpi = compute_logprob_sgp(means, stddevs, action_orgs)

        with tf.GradientTape() as tape:
            entropy_diff = -logpi - self.target_entropy
            log_alpha_loss = tf.reduce_mean(tf.exp(self.log_alpha) * entropy_diff)

        grad = tape.gradient(log_alpha_loss, self.log_alpha)
        self.alpha_optimizer.apply_gradients([(grad, self.log_alpha)])

        # --- soft target update
        self.parameter.q_target.set_weights(
            (1 - self.config.soft_target_update_tau) * np.array(self.parameter.q_target.get_weights(), dtype=object)
            + (self.config.soft_target_update_tau) * np.array(self.parameter.q_online.get_weights(), dtype=object)
        )

        # --- hard target sync
        if self.train_count % self.config.hard_target_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())

        self.train_count += 1
        return {
            "q_loss": q_loss.numpy(),
            "policy_loss": policy_loss.numpy(),
            "alpha_loss": log_alpha_loss.numpy(),
        }


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

    def call_policy(self, state: np.ndarray) -> List[float]:
        self.state = state
        action, mean, _, _ = self.parameter.policy(state.reshape(1, -1))

        if self.training:
            action = action.numpy()[0]
        else:
            # テスト時は平均を使う
            action = mean.numpy()[0]

        # Squashed Gaussian Policy (-1, 1) -> (action range)
        action = (action + 1) / 2
        action = self.config.action_low + action * (self.config.action_high - self.config.action_low)

        self.action = action
        return self.action

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> Dict[str, Union[float, int]]:
        if not self.training:
            return {}

        batch = {
            "state": self.state,
            "action": self.action,
            "next_state": next_state,
            "reward": reward,
            "done": done,
        }
        self.remote_memory.add(batch)

        return {}

    def call_render(self, env: EnvRun) -> None:
        state = self.state.reshape(1, -1)
        action = np.asarray([self.action])
        _, mean, stddev, _ = self.parameter.policy(state)
        mean = mean.numpy()[0][0]
        stddev = stddev.numpy()[0][0]
        q1, q2 = self.parameter.q_online(state, action)
        q1 = q1.numpy()[0][0]
        q2 = q2.numpy()[0][0]

        print(f"mean {mean:.5f}, stddev {stddev:.5f}")
        print(f"q1   {q1:.5f}, q2    {q2:.5f}")
