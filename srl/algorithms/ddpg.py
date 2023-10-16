from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union, cast

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import EnvObservationTypes, RLTypes
from srl.base.rl.algorithms.continuous_action import ContinuousActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.rl.registration import register
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, ExperienceReplayBufferConfig
from srl.rl.models.image_block import ImageBlockConfig
from srl.rl.models.mlp_block import MLPBlockConfig
from srl.rl.models.tf.input_block import InputBlock
from srl.rl.processors.image_processor import ImageProcessor
from srl.rl.schedulers.scheduler import SchedulerConfig

kl = keras.layers

"""
Ref
https://spinningup.openai.com/en/latest/algorithms/ddpg.html

DDPG
    Replay buffer       : o
    Target Network(soft): o
    Target Network(hard): o
    Add action noise    : o
TD3
    Clipped Double Q learning : o
    Target Policy Smoothing   : o
    Delayed Policy Update     : o
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

    lr: float = 0.005  # type: ignore , type OK
    discount: float = 0.9
    soft_target_update_tau: float = 0.02
    hard_target_update_interval: int = 100

    noise_stdev: float = 0.2  # ノイズ用の標準偏差
    target_policy_noise_stddev: float = 0.2  # Target policy ノイズの標準偏差
    target_policy_clip_range: float = 0.5  # Target policy ノイズのclip範囲
    actor_update_interval: int = 2

    def __post_init__(self):
        super().__post_init__()
        self.lr: SchedulerConfig = SchedulerConfig(cast(float, self.lr))

    def set_processor(self) -> List[Processor]:
        return [
            ImageProcessor(
                image_type=EnvObservationTypes.GRAY_2ch,
                resize=(84, 84),
                enable_norm=True,
            )
        ]

    @property
    def base_action_type(self) -> RLTypes:
        return RLTypes.CONTINUOUS

    @property
    def base_observation_type(self) -> RLTypes:
        return RLTypes.CONTINUOUS

    def get_use_framework(self) -> str:
        return "tensorflow"

    def getName(self) -> str:
        return "DDPG"

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
class _ActorNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        # input
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)

        # image
        if self.in_block.use_image_layer:
            self.image_block = config.image_block.create_block_tf(enable_time_distributed_layer=False)
            self.image_flatten = kl.Flatten()

        # --- hidden block
        self.hidden_block = config.policy_hidden_block.create_block_tf()

        # --- out layer
        self.out_layer = kl.Dense(config.action_num, activation="tanh")

        # build
        self.build((None,) + config.observation_shape)

    def call(self, state, training=False):
        x = self.in_block(state, training=training)
        if self.in_block.use_image_layer:
            x = self.image_block(x, training=training)
            x = self.image_flatten(x)
        x = self.hidden_block(x, training=training)
        x = self.out_layer(x, training=training)
        return x

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


class _CriticNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        # input
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)

        # image
        if self.in_block.use_image_layer:
            self.image_block = config.image_block.create_block_tf(False)
            self.image_flatten = kl.Flatten()

        # q1
        self.q1_block = config.q_hidden_block.create_block_tf()
        self.q1_output = kl.Dense(
            1,
            activation="linear",
            kernel_initializer="truncated_normal",
            bias_initializer="truncated_normal",
        )

        # q2
        self.q2_block = config.q_hidden_block.create_block_tf()
        self.q2_output = kl.Dense(
            1,
            activation="linear",
            kernel_initializer="truncated_normal",
            bias_initializer="truncated_normal",
        )

        # build
        self.build(
            [
                (None,) + config.observation_shape,
                (None, config.action_num),
            ]
        )

    def call(self, inputs, training=False):
        state = inputs[0]
        action = inputs[1]

        x = self.in_block(state, training=training)
        if self.in_block.use_image_layer:
            x = self.image_block(x, training=training)
            x = self.image_flatten(x, training=training)
        x = tf.concat([x, action], axis=1)

        # q1
        q1 = self.q1_block(x, training=training)
        q1 = self.q1_output(q1, training=training)
        # q2
        q2 = self.q2_block(x, training=training)
        q2 = self.q2_output(q2, training=training)

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
        model = keras.Model(inputs=x, outputs=self.call(x), name=name)
        model.summary(**kwargs)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.actor_online = _ActorNetwork(self.config)
        self.actor_target = _ActorNetwork(self.config)
        self.critic_online = _CriticNetwork(self.config)
        self.critic_target = _CriticNetwork(self.config)

        self.actor_target.set_weights(self.actor_online.get_weights())
        self.critic_target.set_weights(self.critic_online.get_weights())

    def call_restore(self, data: Any, **kwargs) -> None:
        self.actor_online.set_weights(data[0])
        self.actor_target.set_weights(data[0])
        self.critic_online.set_weights(data[1])
        self.critic_target.set_weights(data[1])

    def call_backup(self, **kwargs) -> Any:
        return [
            self.actor_online.get_weights(),
            self.critic_online.get_weights(),
        ]

    def summary(self, **kwargs):
        self.actor_online.summary(**kwargs)
        self.critic_online.summary(**kwargs)


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_sch = self.config.lr.create_schedulers()
        lr = self.lr_sch.get_rate()

        self.actor_optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=lr)

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

        # ポリシーより次の状態のアクションを取得
        n_actions = self.parameter.actor_target(n_states)

        # Target Actionのノイズ
        clipped_noise = np.clip(
            np.random.normal(0, self.config.target_policy_noise_stddev, size=n_actions.shape),
            -self.config.target_policy_clip_range,
            self.config.target_policy_clip_range,
        )
        n_actions = np.clip(n_actions + clipped_noise, -1, 1)

        # 2つのQ値から小さいほうを採用(Clipped Double Q learning)して、
        # Q値を計算 : reward if done else (reward + discount * n_qval) - (alpha * H)
        n_q1, n_q2 = self.parameter.critic_target([n_states, n_actions])
        q_vals = rewards + (1 - dones) * self.config.discount * tf.minimum(n_q1, n_q2)

        # --- ポリシーの学習
        # Actorの学習は少し減らす
        _info = {}
        if self.train_count % self.config.actor_update_interval == 0:
            with tf.GradientTape() as tape:
                # アクションを出力
                actor_actions = self.parameter.actor_online(states, training=True)
                q, _ = self.parameter.critic_online([states, actor_actions], training=True)
                actor_loss = -tf.reduce_mean(q)  # 最大化
                actor_loss += tf.reduce_sum(self.parameter.actor_online.losses)

            grads = tape.gradient(actor_loss, self.parameter.actor_online.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads, self.parameter.actor_online.trainable_variables))

            _info["actor_loss"] = actor_loss.numpy()

        # --- Qモデルの学習
        with tf.GradientTape() as tape:
            q1, q2 = self.parameter.critic_online([states, actions], training=True)
            loss1 = tf.reduce_mean(tf.square(q_vals - q1))
            loss2 = tf.reduce_mean(tf.square(q_vals - q2))
            critic_loss = (loss1 + loss2) / 2
            critic_loss += tf.reduce_sum(self.parameter.critic_online.losses)

        grads = tape.gradient(critic_loss, self.parameter.critic_online.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.parameter.critic_online.trainable_variables))

        _info["critic_loss"] = critic_loss.numpy()

        # lr_schedule
        if self.lr_sch.update(self.train_count):
            lr = self.lr_sch.get_rate()
            self.actor_optimizer.learning_rate = lr
            self.critic_optimizer.learning_rate = lr
            _info["lr"] = lr

        # --- soft target update
        self.parameter.actor_target.set_weights(
            (1 - self.config.soft_target_update_tau)
            * np.array(self.parameter.actor_target.get_weights(), dtype=object)
            + (self.config.soft_target_update_tau) * np.array(self.parameter.actor_online.get_weights(), dtype=object)
        )
        self.parameter.critic_target.set_weights(
            (1 - self.config.soft_target_update_tau)
            * np.array(self.parameter.critic_target.get_weights(), dtype=object)
            + (self.config.soft_target_update_tau) * np.array(self.parameter.critic_online.get_weights(), dtype=object)
        )

        # --- hard target sync
        if self.train_count % self.config.hard_target_update_interval == 0:
            self.parameter.actor_target.set_weights(self.parameter.actor_online.get_weights())
            self.parameter.critic_target.set_weights(self.parameter.critic_online.get_weights())

        self.train_count += 1
        self.train_info = _info


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
        self.action = self.parameter.actor_online(state.reshape(1, -1)).numpy()[0]  # type:ignore

        if self.training:
            # 学習用はノイズを混ぜる
            noise = np.random.normal(0, self.config.noise_stdev, size=self.config.action_num)
            self.action = np.clip(self.action + noise, -1, 1)

        # (-1, 1) -> (action range)
        env_action = (self.action + 1) / 2
        env_action = self.config.action_low + env_action * (self.config.action_high - self.config.action_low)
        return env_action.tolist(), {}

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
        self.memory.add(batch)

        return {}

    def render_terminal(self, worker, **kwargs) -> None:
        state = self.state.reshape(1, -1)
        action = self.parameter.actor_online(state)
        q1, q2 = self.parameter.critic_online([state, action])  # type:ignore
        q1 = q1.numpy()[0][0]
        q2 = q2.numpy()[0][0]
        action = action.numpy()[0]  # type:ignore
        print(f"action {action}")
        print(f"q1   {q1:.5f}, q2    {q2:.5f}")
