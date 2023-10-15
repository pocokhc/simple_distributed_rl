from dataclasses import dataclass, field
from typing import Any, List, Tuple, cast

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import EnvObservationTypes, RLTypes
from srl.base.rl.algorithms.continuous_action import ContinuousActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.rl.registration import register
from srl.rl.functions.common_tf import compute_logprob_sgp
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, ExperienceReplayBufferConfig
from srl.rl.models.image_block import ImageBlockConfig
from srl.rl.models.mlp_block import MLPBlockConfig
from srl.rl.models.tf.input_block import InputBlock
from srl.rl.processors.image_processor import ImageProcessor
from srl.rl.schedulers.scheduler import SchedulerConfig
from srl.utils.common import compare_less_version

kl = keras.layers

"""
Paper
https://arxiv.org/abs/1812.05905

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
class Config(RLConfig, ExperienceReplayBufferConfig):
    # model
    image_block: ImageBlockConfig = field(init=False, default_factory=lambda: ImageBlockConfig())
    policy_hidden_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())
    q_hidden_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())

    discount: float = 0.9
    lr_policy: float = 0.001  # type: ignore , type OK
    lr_q: float = 0.001  # type: ignore , type OK
    lr_alpha: float = 0.0001  # type: ignore , type OK
    soft_target_update_tau: float = 0.02
    hard_target_update_interval: int = 100

    enable_alpha_train: bool = True
    alpha_no_learning: float = 0.1  # 学習しない場合のalpha値

    #: 勾配爆発の対策, 平均、分散、ランダムアクションで大きい値を出さないようにclipする
    enable_stable_gradients: bool = True  # 勾配爆発の対策
    stable_gradients_max_stddev: float = 1
    stable_gradients_normal_max_action: float = 1

    def __post_init__(self):
        super().__post_init__()

        self.memory.capacity = 1000
        self.lr_policy: SchedulerConfig = SchedulerConfig(cast(float, self.lr_policy))
        self.lr_q: SchedulerConfig = SchedulerConfig(cast(float, self.lr_q))
        self.lr_alpha: SchedulerConfig = SchedulerConfig(cast(float, self.lr_alpha))
        self.policy_hidden_block.set_mlp((64, 64, 64))
        self.q_hidden_block.set_mlp((128, 128, 128))

    @property
    def base_action_type(self) -> RLTypes:
        return RLTypes.CONTINUOUS

    @property
    def base_observation_type(self) -> RLTypes:
        return RLTypes.CONTINUOUS

    def get_use_framework(self) -> str:
        return "tensorflow"

    def set_processor(self) -> List[Processor]:
        return [
            ImageProcessor(
                image_type=EnvObservationTypes.GRAY_2ch,
                resize=(84, 84),
                enable_norm=True,
            )
        ]

    def getName(self) -> str:
        return "SAC"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_memory()


register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


# ------------------------------------------------------
# Memory
# ------------------------------------------------------
class Memory(ExperienceReplayBuffer):
    pass


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _PolicyNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # input
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)

        # image
        if self.in_block.use_image_layer:
            self.image_block = config.image_block.create_block_tf(enable_time_distributed_layer=False)
            self.image_flatten = kl.Flatten()

        # --- hidden block
        self.hidden_block = config.policy_hidden_block.create_block_tf()

        # --- out layer
        self.mean_layer = kl.Dense(
            config.action_num,
            activation="linear",
        )
        # 分散が大きいとinfになるので0-1あたりに抑える
        self.stddev_layer = kl.Dense(
            config.action_num,
            activation="linear",
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.5),
            bias_initializer="zeros",
        )

        # build
        self.build((None,) + config.observation_shape)

    def call(self, state, training=False):
        x = self.in_block(state, training=training)
        if self.in_block.use_image_layer:
            x = self.image_block(x, training=training)
            x = self.image_flatten(x)
        x = self.hidden_block(x, training=training)
        mean = self.mean_layer(x)
        stddev = self.stddev_layer(x)

        # σ > 0
        stddev = tf.exp(stddev)

        if self.config.enable_stable_gradients:
            stddev = tf.clip_by_value(stddev, 0, self.config.stable_gradients_max_stddev)

        if mean.shape[0] is None:
            return mean, stddev

        # Reparameterization trick
        normal_random = tf.random.normal(mean.shape, mean=0.0, stddev=1.0)
        if self.config.enable_stable_gradients:
            normal_random = tf.clip_by_value(
                normal_random,
                -self.config.stable_gradients_normal_max_action,
                self.config.stable_gradients_normal_max_action,
            )
        action = mean + stddev * normal_random

        # Squashed Gaussian Policy
        sgp_action = tf.tanh(action)

        return sgp_action, mean, stddev, action

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def summary(self, name: str = "", **kwargs):
        if hasattr(self.in_block, "init_model_graph"):
            self.in_block.init_model_graph()
        if self.in_block.use_image_layer and hasattr(self.image_block, "init_model_graph"):
            self.image_block.init_model_graph()
        if hasattr(self.hidden_block, "init_model_graph"):
            self.hidden_block.init_model_graph()

        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x), name=name)
        model.summary(**kwargs)


class _DualQNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        # input
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)

        # image
        if self.in_block.use_image_layer:
            self.image_block = config.image_block.create_block_tf(enable_time_distributed_layer=False)
            self.image_flatten = kl.Flatten()

        # q1
        self.q1_block = config.q_hidden_block.create_block_tf()
        self.q1_out_layer = kl.Dense(1)

        # q2
        self.q2_block = config.q_hidden_block.create_block_tf()
        self.q2_out_layer = kl.Dense(1)

        # build
        self.build(
            [
                (None,) + config.observation_shape,
                (None, config.action_num),
            ]
        )

    @tf.function
    def call(self, x, training=False):
        return self._call(x, training)

    def _call(self, inputs, training=False):
        state = inputs[0]
        action = inputs[1]

        x = self.in_block(state, training=training)
        if self.in_block.use_image_layer:
            x = self.image_block(x, training=training)
            x = self.image_flatten(x)
        x = tf.concat([x, action], axis=1)

        # q1
        q1 = self.q1_block(x, training=training)
        q1 = self.q1_out_layer(q1, training=training)

        # q2
        q2 = self.q2_block(x, training=training)
        q2 = self.q2_out_layer(q2, training=training)

        return q1, q2

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def summary(self, name: str = "", **kwargs):
        if hasattr(self.in_block, "init_model_graph"):
            self.in_block.init_model_graph()
        if self.in_block.use_image_layer and hasattr(self.image_block, "init_model_graph"):
            self.image_block.init_model_graph()
        if hasattr(self.q1_block, "init_model_graph"):
            self.q1_block.init_model_graph()
        if hasattr(self.q2_block, "init_model_graph"):
            self.q2_block.init_model_graph()

        x = [
            kl.Input(shape=self.__input_shape[0][1:]),
            kl.Input(shape=self.__input_shape[1][1:]),
        ]
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self._call(x), name=name)
        model.summary(**kwargs)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.policy = _PolicyNetwork(self.config)
        self.q_online = _DualQNetwork(self.config)
        self.q_target = _DualQNetwork(self.config)
        self.q_target.set_weights(self.q_online.get_weights())

    def call_restore(self, data: Any, **kwargs) -> None:
        self.policy.set_weights(data[0])
        self.q_online.set_weights(data[1])
        self.q_target.set_weights(data[1])

    def call_backup(self, **kwargs) -> Any:
        return [
            self.policy.get_weights(),
            self.q_online.get_weights(),
        ]

    def summary(self, **kwargs):
        self.policy.summary(**kwargs)
        self.q_online.summary(**kwargs)


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_policy_sch = self.config.lr_policy.create_schedulers()
        self.lr_q_sch = self.config.lr_q.create_schedulers()
        self.lr_alpha_sch = self.config.lr_alpha.create_schedulers()

        if compare_less_version(tf.__version__, "2.11.0"):
            self.q_optimizer = keras.optimizers.Adam(learning_rate=self.lr_policy_sch.get_rate(0))
            self.policy_optimizer = keras.optimizers.Adam(learning_rate=self.lr_q_sch.get_rate(0))
            self.alpha_optimizer = keras.optimizers.Adam(learning_rate=self.lr_alpha_sch.get_rate(0))
        else:
            self.q_optimizer = keras.optimizers.legacy.Adam(learning_rate=self.lr_policy_sch.get_rate(0))
            self.policy_optimizer = keras.optimizers.legacy.Adam(learning_rate=self.lr_q_sch.get_rate(0))
            self.alpha_optimizer = keras.optimizers.legacy.Adam(learning_rate=self.lr_alpha_sch.get_rate(0))

        # エントロピーαの目標値、-1×アクション数が良いらしい
        self.target_entropy = -1 * self.config.action_num

        # エントロピーα自動調整用
        self.log_alpha = tf.Variable(0.5, dtype=tf.float32)

    def train_on_batchs(self, memory_sample_return) -> None:
        batchs = memory_sample_return

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
        if self.config.enable_alpha_train:
            alpha = tf.math.exp(self.log_alpha)
        else:
            alpha = self.config.alpha_no_learning

        # ポリシーより次の状態のアクションを取得
        n_actions, n_means, n_stddevs, n_action_orgs = self.parameter.policy(n_states)
        # 次の状態のアクションのlogpiを取得
        n_logpi = compute_logprob_sgp(n_means, n_stddevs, n_action_orgs)

        # 2つのQ値から小さいほうを採用(Clipped Double Q learning)して、
        # Q値を計算 : reward if done else (reward + discount * n_qval) - (alpha * H)
        n_q1, n_q2 = self.parameter.q_target([n_states, n_actions])
        n_qval = self.config.discount * tf.minimum(n_q1, n_q2)
        q_vals = rewards + (1 - dones) * n_qval - (alpha * n_logpi)

        # --- Qモデルの学習
        with tf.GradientTape() as tape:
            q1, q2 = self.parameter.q_online([states, actions], training=True)
            loss1 = tf.reduce_mean(tf.square(q_vals - q1))
            loss2 = tf.reduce_mean(tf.square(q_vals - q2))
            q_loss = (loss1 + loss2) / 2
            q_loss += tf.reduce_sum(self.parameter.q_online.losses)  # 正則化項

        grads = tape.gradient(q_loss, self.parameter.q_online.trainable_variables)
        self.q_optimizer.apply_gradients(zip(grads, self.parameter.q_online.trainable_variables))

        # --- ポリシーの学習
        with tf.GradientTape() as tape:
            # アクションを出力
            selected_actions, means, stddevs, action_orgs = self.parameter.policy(states, training=True)

            # logπ(a|s)
            logpi = compute_logprob_sgp(means, stddevs, action_orgs)

            # Q値を出力、小さいほうを使う
            q1, q2 = self.parameter.q_online([states, selected_actions], training=True)
            q_min = tf.minimum(q1, q2)

            # alphaは定数扱いなので勾配が流れないようにする
            policy_loss = q_min - (tf.stop_gradient(alpha) * logpi)
            policy_loss = -tf.reduce_mean(policy_loss)
            policy_loss += tf.reduce_sum(self.parameter.policy.losses)  # 正則化項

        grads = tape.gradient(policy_loss, self.parameter.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(grads, self.parameter.policy.trainable_variables))

        # --- 方策エントロピーαの自動調整
        if self.config.enable_alpha_train:
            _, means, stddevs, action_orgs = self.parameter.policy(states)
            logpi = compute_logprob_sgp(means, stddevs, action_orgs)

            with tf.GradientTape() as tape:
                entropy_diff = logpi + self.target_entropy
                log_alpha_loss = tf.reduce_mean(-tf.exp(self.log_alpha) * entropy_diff)

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

        self.q_optimizer.learning_rate = self.lr_q_sch.get_rate(self.train_count)
        self.policy_optimizer.learning_rate = self.lr_policy_sch.get_rate(self.train_count)
        self.alpha_optimizer.learning_rate = self.lr_alpha_sch.get_rate(self.train_count)

        self.train_count += 1
        self.train_info = {
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
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

    def call_on_reset(self, state: np.ndarray) -> dict:
        self.state = state
        return {}

    def call_policy(self, state: np.ndarray) -> Tuple[List[float], dict]:
        self.state = state
        action, mean, stddev, _ = self.parameter.policy(state.reshape(1, -1))  # type:ignore , ignore check "None"

        if self.training:
            action = action.numpy()[0]
        else:
            # テスト時は平均を使う
            mean = tf.tanh(mean)
            action = mean.numpy()[0]

        # Squashed Gaussian Policy (-1, 1) -> (action range)
        env_action = (action + 1) / 2
        env_action = self.config.action_low + env_action * (self.config.action_high - self.config.action_low)

        self.mean = mean.numpy()[0]
        self.stddev = stddev.numpy()[0]
        self.action = action
        return env_action.tolist(), {}

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> dict:
        if not self.training:
            return {}

        batch = {
            "state": self.state,
            "action": self.action,
            "next_state": next_state,
            "reward": reward,
            "done": done,
        }
        self.memory.add(batch)

        return {}

    def render_terminal(self, worker, **kwargs) -> None:
        q1, q2 = self.parameter.q_online([self.state.reshape(1, -1), np.asarray([self.action])])
        q1 = q1.numpy()[0][0]
        q2 = q2.numpy()[0][0]

        print(f"mean   {self.mean}")
        print(f"stddev {self.stddev}")
        print(f"q1   {q1:.5f}, q2    {q2:.5f}")
