from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import InfoType, RLBaseTypes
from srl.base.rl.config import RLConfig
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import ObservationProcessor
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker import RLWorker
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.box import BoxSpace
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, RLConfigComponentExperienceReplayBuffer
from srl.rl.models.config.framework_config import RLConfigComponentFramework
from srl.rl.models.config.mlp_block import MLPBlockConfig
from srl.rl.models.tf.blocks.input_block import create_in_block_out_value
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
class Config(
    RLConfig[ArrayContinuousSpace, BoxSpace],
    RLConfigComponentExperienceReplayBuffer,
    RLConfigComponentFramework,
):
    """
    <:ref:`RLConfigComponentExperienceReplayBuffer`>
    <:ref:`RLConfigComponentFramework`>
    """

    #: <:ref:`MLPBlock`> policy layers
    policy_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())
    #: <:ref:`MLPBlock`> q layers
    q_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())

    #: <:ref:`scheduler`> Learning rate
    lr: Union[float, SchedulerConfig] = 0.005
    #: discount
    discount: float = 0.9
    #: soft_target_update_tau
    soft_target_update_tau: float = 0.02
    #: hard_target_update_interval
    hard_target_update_interval: int = 100

    #: ノイズ用の標準偏差
    noise_stddev: float = 0.2
    #: Target policy ノイズの標準偏差
    target_policy_noise_stddev: float = 0.2
    #: Target policy ノイズのclip範囲
    target_policy_clip_range: float = 0.5
    #: Actorの学習間隔
    actor_update_interval: int = 2

    def __post_init__(self):
        super().__post_init__()

    def get_processors(self) -> List[Optional[ObservationProcessor]]:
        return [self.input_image_block.get_processor()]

    def get_base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.CONTINUOUS

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.CONTINUOUS

    def get_framework(self) -> str:
        return "tensorflow"

    def get_name(self) -> str:
        return "DDPG"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_memory()
        self.assert_params_framework()


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

        # --- input
        self.input_block = create_in_block_out_value(
            config.input_value_block,
            config.input_image_block,
            config.observation_space,
        )

        # --- hidden block
        self.hidden_block = config.policy_block.create_block_tf()

        # --- out layer
        self.out_layer = kl.Dense(config.action_space.size, activation="tanh")

        # build
        self.build((None,) + config.observation_space.shape)

    def call(self, x, training=False):
        x = self.input_block(x, training)
        x = self.hidden_block(x, training=training)
        x = self.out_layer(x, training=training)
        return x

    @tf.function
    def compute_train_loss(self, critic_model, state):
        action = self(state, training=True)
        q1, q2 = critic_model([state, action])
        # q1を最大化するように方策を更新
        loss = -tf.reduce_mean(q1)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss


class _CriticNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        self.input_block = create_in_block_out_value(
            config.input_value_block,
            config.input_image_block,
            config.observation_space,
        )

        # q1
        self.q1_block = config.q_block.create_block_tf()
        self.q1_output = kl.Dense(
            1,
            activation="linear",
            kernel_initializer="truncated_normal",
            bias_initializer="truncated_normal",
        )

        # q2
        self.q2_block = config.q_block.create_block_tf()
        self.q2_output = kl.Dense(
            1,
            activation="linear",
            kernel_initializer="truncated_normal",
            bias_initializer="truncated_normal",
        )

        # build
        self.build(
            [
                (None,) + config.observation_space.shape,
                (None, config.action_space.size),
            ]
        )

    def call(self, inputs, training=False):
        x = inputs[0]
        action = inputs[1]

        x = self.input_block(x, training)
        x = tf.concat([x, action], axis=1)

        # q1
        q1 = self.q1_block(x, training=training)
        q1 = self.q1_output(q1, training=training)
        # q2
        q2 = self.q2_block(x, training=training)
        q2 = self.q2_output(q2, training=training)

        return q1, q2

    @tf.function
    def compute_train_loss(self, state, action, target_q):
        q1, q2 = self([state, action], training=True)
        loss1 = tf.reduce_mean(tf.square(target_q - q1))
        loss2 = tf.reduce_mean(tf.square(target_q - q2))
        loss = (loss1 + loss2) / 2
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter[Config]):
    def __init__(self, *args):
        super().__init__(*args)

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
class Trainer(RLTrainer[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)

        self.lr_sch = SchedulerConfig.create_scheduler(self.config.lr)
        lr = self.lr_sch.get_rate()

        self.actor_optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=lr)

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        batchs = self.memory.sample(self.batch_size)
        self.info = {}

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
        target_q = rewards + (1 - dones) * self.config.discount * tf.minimum(n_q1, n_q2)

        # --- ポリシーの学習
        # Actorの学習は少し減らす
        if self.train_count % self.config.actor_update_interval == 0:
            self.parameter.critic_online.trainable = False
            with tf.GradientTape() as tape:
                actor_loss = self.parameter.actor_online.compute_train_loss(self.parameter.critic_online, states)
            grads = tape.gradient(actor_loss, self.parameter.actor_online.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads, self.parameter.actor_online.trainable_variables))
            self.info["actor_loss"] = actor_loss.numpy()

        # --- Qモデルの学習
        self.parameter.critic_online.trainable = True
        with tf.GradientTape() as tape:
            critic_loss = self.parameter.critic_online.compute_train_loss(states, actions, target_q)
        grads = tape.gradient(critic_loss, self.parameter.critic_online.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.parameter.critic_online.trainable_variables))
        self.info["critic_loss"] = critic_loss.numpy()

        # lr_schedule
        if self.lr_sch.update(self.train_count):
            lr = self.lr_sch.get_rate()
            self.actor_optimizer.learning_rate = lr
            self.critic_optimizer.learning_rate = lr
            self.info["lr"] = lr

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


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)

    def on_reset(self, worker) -> InfoType:
        return {}

    def policy(self, worker) -> Tuple[List[float], InfoType]:
        self.action = self.parameter.actor_online(worker.state[np.newaxis, ...]).numpy()[0]

        if self.training:
            # 学習用はノイズを混ぜる
            noise = np.random.normal(0, self.config.noise_stddev, size=self.config.action_space.size)
            self.action = np.clip(self.action + noise, -1, 1)

        # (-1, 1) -> (action range)
        env_action = (self.action + 1) / 2
        env_action = self.config.action_space.low + env_action * (
            self.config.action_space.high - self.config.action_space.low
        )
        env_action = env_action.tolist()
        return env_action, {}

    def on_step(self, worker) -> InfoType:
        if not self.training:
            return {}

        batch = {
            "state": worker.prev_state,
            "action": self.action,
            "next_state": worker.state,
            "reward": worker.reward,
            "done": worker.terminated,
        }
        self.memory.add(batch)

        return {}

    def render_terminal(self, worker, **kwargs) -> None:
        state = worker.prev_state.reshape(1, -1)
        action = self.parameter.actor_online(state)
        q1, q2 = self.parameter.critic_online([state, action])
        q1 = q1.numpy()[0][0]
        q2 = q2.numpy()[0][0]
        action = action.numpy()[0]
        print(f"action {action}")
        print(f"q1   {q1:.5f}, q2    {q2:.5f}")
