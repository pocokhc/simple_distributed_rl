from dataclasses import dataclass
from typing import Any, Dict, List, Union, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kl
from srl.base.define import EnvObservationType, RLObservationType
from srl.base.rl.algorithms.continuous_action import ContinuousActionConfig, ContinuousActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.processor import Processor
from srl.base.rl.processors.image_processor import ImageProcessor
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import ExperienceReplayBuffer
from srl.rl.models.dqn_image_block import DQNImageBlock
from srl.rl.models.input_layer import create_input_layer
from srl.rl.models.mlp_block import MLPBlock

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
class Config(ContinuousActionConfig):

    # model
    cnn_block: kl.Layer = DQNImageBlock
    cnn_block_kwargs: dict = None
    policy_hidden_block: kl.Layer = MLPBlock
    policy_hidden_block_kwargs: dict = None
    q_hidden_block: kl.Layer = MLPBlock
    q_hidden_block_kwargs: dict = None

    discount: float = 0.9  # 割引率
    lr: float = 0.005  # 学習率
    soft_target_update_tau: float = 0.02
    hard_target_update_interval: int = 100

    batch_size: int = 32
    capacity: int = 100_000
    memory_warmup_size: int = 1000

    noise_stdev: float = 0.2  # ノイズ用の標準偏差
    target_policy_noise_stddev: float = 0.2  # Target policy ノイズの標準偏差
    target_policy_clip_range: float = 0.5  # Target policy ノイズのclip範囲
    actor_update_interval: int = 2

    def __post_init__(self):
        super().__init__()
        if self.cnn_block_kwargs is None:
            self.cnn_block_kwargs = {}
        if self.policy_hidden_block_kwargs is None:
            self.policy_hidden_block_kwargs = {}
        if self.q_hidden_block_kwargs is None:
            self.q_hidden_block_kwargs = {}

    def set_processor(self) -> List[Processor]:
        return [
            ImageProcessor(
                image_type=EnvObservationType.GRAY_2ch,
                resize=(84, 84),
                enable_norm=True,
            )
        ]

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.CONTINUOUS

    @staticmethod
    def getName() -> str:
        return "DDPG"

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
class _ActorNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        in_state, c, use_image_head = create_input_layer(
            config.observation_shape,
            config.env_observation_type,
        )
        if use_image_head:
            c = config.cnn_block(**config.cnn_block_kwargs)(c)
            c = kl.Flatten()(c)

        # --- hidden block
        c = config.policy_hidden_block(**config.policy_hidden_block_kwargs)(c)

        # --- out layer
        c = kl.Dense(config.action_num, activation="tanh")(c)
        self.model = keras.Model(in_state, c, name="ActorNetwork")

        # 重みを初期化
        dummy_state = np.zeros(shape=(1,) + config.observation_shape, dtype=np.float32)
        actions = self(dummy_state)
        assert actions.shape == (1, config.action_num)

    def call(self, state):
        return self.model(state)


class _CriticNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        # in state
        in_state, c, use_image_head = create_input_layer(
            config.observation_shape,
            config.env_observation_type,
        )
        if use_image_head:
            c = config.cnn_block(**config.cnn_block_kwargs)(c)
            c = kl.Flatten()(c)

        # in action
        in_action = kl.Input(shape=(config.action_num,))
        c = kl.Concatenate()([c, in_action])

        # q1
        c1 = config.q_hidden_block(**config.q_hidden_block_kwargs)(c)
        q1 = kl.Dense(
            1, activation="linear", kernel_initializer="truncated_normal", bias_initializer="truncated_normal"
        )(c1)

        # q2
        c2 = config.q_hidden_block(**config.q_hidden_block_kwargs)(c)
        q2 = kl.Dense(
            1, activation="linear", kernel_initializer="truncated_normal", bias_initializer="truncated_normal"
        )(c2)

        # out layer
        self.model = keras.Model([in_state, in_action], [q1, q2], name="CriticNetwork")

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

        self.actor_online = _ActorNetwork(self.config)
        self.actor_target = _ActorNetwork(self.config)
        self.critic_online = _CriticNetwork(self.config)
        self.critic_target = _CriticNetwork(self.config)

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
        self.actor_online.model.summary(**kwargs)
        self.critic_online.model.summary(**kwargs)


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

        self.actor_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr)

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
        n_q1, n_q2 = self.parameter.critic_target(n_states, n_actions)
        q_vals = rewards + (1 - dones) * self.config.discount * tf.minimum(n_q1, n_q2)

        # --- ポリシーの学習
        # Actorの学習は少し減らす
        if self.train_count % self.config.actor_update_interval == 0:
            with tf.GradientTape() as tape:
                # アクションを出力
                actor_actions = self.parameter.actor_online(states)
                q, _ = self.parameter.critic_online(states, actor_actions)
                actor_loss = -tf.reduce_mean(q)  # 最大化

            grads = tape.gradient(actor_loss, self.parameter.actor_online.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads, self.parameter.actor_online.trainable_variables))

            actor_loss = actor_loss.numpy()
        else:
            actor_loss = None

        # --- Qモデルの学習
        with tf.GradientTape() as tape:
            q1, q2 = self.parameter.critic_online(states, actions)
            loss1 = tf.reduce_mean(tf.square(q_vals - q1))
            loss2 = tf.reduce_mean(tf.square(q_vals - q2))
            critic_loss = (loss1 + loss2) / 2

        grads = tape.gradient(critic_loss, self.parameter.critic_online.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.parameter.critic_online.trainable_variables))

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
        return {
            "critic_loss": critic_loss.numpy(),
            "actor_loss": actor_loss,
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
        self.action = self.parameter.actor_online(state.reshape(1, -1)).numpy()[0]

        if self.training:
            # 学習用はノイズを混ぜる
            noise = np.random.normal(0, self.config.noise_stdev, size=self.config.action_num)
            self.action = np.clip(self.action + noise, -1, 1)

        # (-1, 1) -> (action range)
        env_action = (self.action + 1) / 2
        env_action = self.config.action_low + env_action * (self.config.action_high - self.config.action_low)
        return env_action

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

    def render_terminal(self, env, worker, **kwargs) -> None:
        state = self.state.reshape(1, -1)
        action = self.parameter.actor_online(state)
        q1, q2 = self.parameter.critic_online(state, action)
        q1 = q1.numpy()[0][0]
        q2 = q2.numpy()[0][0]
        action = action.numpy()[0]
        print(f"action {action}")
        print(f"q1   {q1:.5f}, q2    {q2:.5f}")
