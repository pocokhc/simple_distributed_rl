import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.algorithms.base_dqn import RLConfig, RLWorker
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import RLProcessor
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.base.spaces.space import SpaceBase
from srl.rl.memories.replay_buffer import ReplayBuffer, ReplayBufferConfig
from srl.rl.models.config.input_image_block import InputImageBlockConfig
from srl.rl.models.config.input_value_block import InputValueBlockConfig
from srl.rl.models.config.mlp_block import MLPBlockConfig
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig
from srl.rl.schedulers.scheduler import SchedulerConfig
from srl.rl.tf.model import KerasModelAddedSummary

kl = keras.layers


"""
Paper: https://arxiv.org/abs/2302.11563

image augmentation : x
SND-V   : o
SND-STD : x
SND-VIC : x
"""


@dataclass
class Config(RLConfig):
    #: ε-greedy parameter for Test
    test_epsilon: float = 0
    #: ε-greedy parameter for Train
    epsilon: float = 0.001
    #: <:ref:`SchedulerConfig`>
    epsilon_scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig())
    #: Learning rate
    lr: float = 0.001
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())

    #: Batch size
    batch_size: int = 32
    #: <:ref:`ReplayBufferConfig`>
    memory: ReplayBufferConfig = field(default_factory=lambda: ReplayBufferConfig())

    #: Discount rate
    discount: float = 0.99
    #: Synchronization interval to Target network
    target_model_update_interval: int = 1000

    #: int reward scale
    int_reward_scale: float = 0.5

    #: <:ref:`InputValueBlockConfig`>
    input_value_block: InputValueBlockConfig = field(default_factory=lambda: InputValueBlockConfig())
    #: <:ref:`InputImageBlockConfig`>
    input_image_block: InputImageBlockConfig = field(default_factory=lambda: InputImageBlockConfig())
    #: <:ref:`MLPBlockConfig`> hidden layer
    hidden_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())

    def get_name(self) -> str:
        return "SND"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if prev_observation_space.is_image():
            return self.input_image_block.get_processors()
        return []

    def get_framework(self) -> str:
        return "tensorflow"

    def validate_params(self) -> None:
        super().validate_params()
        if not (self.batch_size % 2 == 0):
            raise ValueError(f"assert {self.batch_size} % 2 == 0")


register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


class Memory(RLMemory[Config]):
    def setup(self) -> None:
        self.memory_snd = ReplayBuffer(self.config.memory, self.config.batch_size)
        self.memory_q = ReplayBuffer(self.config.memory, self.config.batch_size)

        self.register_worker_func(self.add_snd, self.memory_snd.serialize)
        self.register_worker_func(self.add_q, self.memory_q.serialize)
        self.register_trainer_recv_func(self.sample_snd)
        self.register_trainer_recv_func(self.sample_q)

    def length(self):
        return self.memory_q.length() + self.memory_snd.length()

    def add_snd(self, batch, serialized: bool = False) -> None:
        self.memory_snd.add(batch, serialized)

    def add_q(self, batch, serialized: bool = False) -> None:
        self.memory_q.add(batch, serialized)

    def sample_snd(self):
        if self.memory_snd.is_warmup_needed():
            return None
        return [
            self.memory_snd.sample(),
            self.memory_snd.sample(batch_size=int(self.config.batch_size / 2)),
        ]

    def sample_q(self):
        return self.memory_q.sample()

    def call_backup(self, **kwargs):
        return [
            self.memory_snd.call_backup(),
            self.memory_q.call_backup(),
        ]

    def call_restore(self, data, **kwargs) -> None:
        self.memory_snd.call_restore(data[0])
        self.memory_q.call_restore(data[1])


class SNDNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        if config.observation_space.is_value():
            self.in_block = config.input_value_block.create_tf_block(config.observation_space)
        elif config.observation_space.is_image():
            self.in_block = config.input_image_block.create_tf_block(config.observation_space)
        else:
            raise ValueError(config.observation_space)

        self.hidden_block = config.hidden_block.create_tf_block()

        # build
        self(self.in_block.create_dummy_data(config.dtype))

        self.loss_mse = keras.losses.MeanSquaredError()

    def call(self, x, training=False):
        x = self.in_block(x, training=training)
        x = self.hidden_block(x, training=training)
        return x

    @tf.function
    def compute_target_train_loss(self, state1, state2, tau):
        # L2ノルムの2乗
        z1 = self(state1, training=True)
        z2 = self(state2, training=True)
        loss = tau - tf.reduce_sum((z1 - z2) ** 2, axis=-1, keepdims=True)
        loss = tf.reduce_sum(loss**2)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss

    @tf.function
    def compute_predictor_train_loss(self, z1, state):
        z2 = self(state, training=True)
        loss = self.loss_mse(z1, z2)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss


class QNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        if config.observation_space.is_value():
            self.in_block = config.input_value_block.create_tf_block(config.observation_space)
        elif config.observation_space.is_image():
            self.in_block = config.input_image_block.create_tf_block(config.observation_space)
        else:
            raise ValueError(config.observation_space)

        self.hidden_block = config.hidden_block.create_tf_block()
        self.out_layer = kl.Dense(config.action_space.n, kernel_initializer="truncated_normal")

        # build
        self(self.in_block.create_dummy_data(config.dtype))

        self.loss_func = keras.losses.Huber()

    def call(self, x, training=False):
        x = self.in_block(x, training=training)
        x = self.hidden_block(x, training=training)
        x = self.out_layer(x)
        return x

    @tf.function
    def compute_train_loss(self, state, onehot_action, target_q):
        q = self(state, training=True)
        q = tf.reduce_sum(q * onehot_action, axis=1)
        loss = self.loss_func(target_q, q)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss


class Parameter(RLParameter[Config]):
    def setup(self):
        self.q_online = QNetwork(self.config, name="Q_online")
        self.q_target = QNetwork(self.config, name="Q_target")
        self.q_target.set_weights(self.q_online.get_weights())

        self.snd_target = SNDNetwork(self.config, name="target")
        self.snd_predictor = SNDNetwork(self.config, name="predictor")

    def call_restore(self, data, **kwargs):
        self.q_online.set_weights(data[0])
        self.q_target.set_weights(data[0])
        self.snd_target.set_weights(data[1])
        self.snd_predictor.set_weights(data[2])

    def call_backup(self, **kwargs):
        return [
            self.q_online.get_weights(),
            self.snd_target.get_weights(),
            self.snd_predictor.get_weights(),
        ]

    def summary(self, **kwargs):
        self.q_online.summary(**kwargs)
        self.snd_target.summary(**kwargs)


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.opt_q = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))
        self.opt_snd_target = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))
        self.opt_snd_predictor = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))
        self.sync_count = 0

    def train(self) -> None:
        self._train_snd()
        self._train_q()

    def _train_snd(self):
        batches = self.memory.sample_snd()
        if batches is None:
            return
        state1, state2 = batches

        # 対照学習
        # (s1, s1) -> tau=0
        # (s1, s2) -> tau=1
        batch_half = int(self.config.batch_size / 2)
        state1 = np.asarray(state1, self.config.dtype)
        state2 = np.asarray(state2, self.config.dtype)

        # random convolution filter skip...
        # random tile mask skip...
        # random
        state1 += np.random.uniform(-0.2, 0.2, size=state1.shape)
        state2 += np.random.uniform(-0.2, 0.2, size=state2.shape)

        state2 = np.concatenate([state1[:batch_half], state2], axis=0)
        tau = np.concatenate(
            [np.zeros((batch_half, 1)), np.ones((batch_half, 1))],
            axis=0,
            dtype=self.config.dtype,
        )

        with tf.GradientTape() as tape:
            loss = self.parameter.snd_target.compute_target_train_loss(state1, state2, tau)
        grad = tape.gradient(loss, self.parameter.snd_target.trainable_variables)
        self.opt_snd_target.apply_gradients(zip(grad, self.parameter.snd_target.trainable_variables))
        self.info["loss_snd_target"] = loss.numpy()

        self.train_count += 1

    def _train_q(self):
        batches = self.memory.sample_q()
        if batches is None:
            return
        state, n_state, action, reward, undone = zip(*batches)
        state = np.asarray(state, self.config.dtype)
        n_state = np.asarray(n_state, self.config.dtype)
        action = np.asarray(action, self.config.dtype)
        reward = np.array(reward, self.config.dtype)
        undone = np.array(undone, self.config.dtype)

        # --- distillation
        z1 = self.parameter.snd_target(n_state)
        with tf.GradientTape() as tape:
            loss = self.parameter.snd_predictor.compute_predictor_train_loss(z1, n_state)
        grad = tape.gradient(loss, self.parameter.snd_predictor.trainable_variables)
        self.opt_snd_predictor.apply_gradients(zip(grad, self.parameter.snd_predictor.trainable_variables))
        self.info["loss_snd_predictor"] = loss.numpy()

        # --- calc next q
        n_q = self.parameter.q_online(n_state)
        n_q_target = self.parameter.q_target(n_state).numpy()
        n_act_idx = np.argmax(n_q, axis=-1)
        maxq = n_q_target[np.arange(self.config.batch_size), n_act_idx]
        target_q = reward + undone * self.config.discount * maxq
        target_q = target_q[..., np.newaxis]

        # --- train q
        with tf.GradientTape() as tape:
            loss = self.parameter.q_online.compute_train_loss(state, action, target_q)
        grad = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.opt_q.apply_gradients(zip(grad, self.parameter.q_online.trainable_variables))
        self.info["loss_q"] = loss.numpy()

        # ------------------------
        # --- targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())
            self.sync_count += 1
            self.info["sync"] = self.sync_count

        self.train_count += 1


class Worker(RLWorker[Config, Parameter, Memory]):
    def on_reset(self, worker):
        self.memory.add_snd(worker.state)

    def policy(self, worker) -> int:
        invalid_actions = worker.invalid_actions

        if self.training:
            epsilon = self.config.epsilon
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            # epsilonより低いならランダム
            action = random.choice([a for a in range(self.config.action_space.n) if a not in invalid_actions])
        else:
            q = self.parameter.q_online(worker.state[np.newaxis, ...])[0].numpy()
            q[invalid_actions] = -np.inf
            action = int(np.argmax(q))

        self.info["epsilon"] = epsilon
        return action

    def on_step(self, worker):
        if not self.training:
            return

        r_ext = worker.reward
        r_int = self._calc_intrinsic_reward(worker.next_state)
        reward = r_ext + self.config.int_reward_scale * r_int

        self.memory.add_q(
            [
                worker.state,
                worker.next_state,
                worker.get_onehot_action(),
                reward,
                int(not worker.terminated),
            ]
        )
        self.memory.add_snd(worker.state)

    def _calc_intrinsic_reward(self, state):
        state = state[np.newaxis, ...]
        z1 = self.parameter.snd_target(state)[0]
        z2 = self.parameter.snd_predictor(state)[0]

        # L2ノルムの2乗
        distance = np.sum(np.square(z1 - z2))

        return distance

    def render_terminal(self, worker, **kwargs) -> None:
        # policy -> render -> env.step -> on_step

        # --- int reward
        r_int = self._calc_intrinsic_reward(worker.state)
        print(f"intrinsic reward: {r_int:.6f}")

        q = self.parameter.q_online(worker.state[np.newaxis, ...])[0]
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        worker.print_discrete_action_info(int(maxa), _render_sub)
