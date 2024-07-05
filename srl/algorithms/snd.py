import random
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.algorithms.base_dqn import RLConfig, RLWorker
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import RLProcessor
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.rl import functions as funcs
from srl.rl.memories.experience_replay_buffer import RandomMemory
from srl.rl.models.config.framework_config import RLConfigComponentFramework
from srl.rl.models.config.mlp_block import MLPBlockConfig
from srl.rl.schedulers.scheduler import SchedulerConfig
from srl.rl.tf.blocks.input_block import create_in_block_out_value

"""
Paper: https://arxiv.org/abs/2302.11563

image augmentation : x
SND-V   : o
SND-STD : x
SND-VIC : x
"""


@dataclass
class Config(
    RLConfig,
    RLConfigComponentFramework,
):
    """
    <:ref:`RLConfigComponentFramework`>
    """

    #: ε-greedy parameter for Test
    test_epsilon: float = 0
    #: ε-greedy parameter for Train
    epsilon: float = 0.001
    #: <:ref:`scheduler`> Learning rate
    lr: Union[float, SchedulerConfig] = 0.001

    #: Batch size
    batch_size: int = 32
    #: capacity
    memory_capacity: int = 100_000
    #: warmup_size
    memory_warmup_size: int = 1_000

    #: Discount rate
    discount: float = 0.99
    #: Synchronization interval to Target network
    target_model_update_interval: int = 1000

    #: int reward scale
    int_reward_scale: float = 0.5

    #: <:ref:`MLPBlock`> hidden layer
    hidden_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())

    def get_processors(self) -> List[Optional[RLProcessor]]:
        return [self.input_image_block.get_processor()]

    def get_framework(self) -> str:
        return "tensorflow"

    def get_name(self) -> str:
        return "SND"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_framework()
        assert self.batch_size % 2 == 0


register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


class Memory(RLMemory[Config]):
    def __init__(self, *args):
        super().__init__(*args)

        self.memory_snd = RandomMemory(self.config.memory_capacity)
        self.memory_q = RandomMemory(self.config.memory_capacity)

    def length(self):
        return self.memory_q.length()

    def add(self, mode: str, batch) -> None:
        if mode == "snd":
            self.memory_snd.add(batch)
        else:
            self.memory_q.add(batch)

    def sample_snd(self, batch_size):
        return self.memory_snd.sample(batch_size)

    def sample_q(self):
        return self.memory_q.sample(self.config.batch_size)


class SNDNetwork(keras.Model):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        self.input_block = create_in_block_out_value(
            config.input_value_block,
            config.input_image_block,
            config.observation_space,
        )

        self.hidden_block = config.hidden_block.create_block_tf(config.action_space.n)

        # build
        self(np.zeros(self.input_block.create_batch_shape((1,)), config.dtype))

        self.loss_mse = keras.losses.MeanSquaredError()

    def call(self, x, training=False):
        x = self.input_block(x, training=training)
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


class QNetwork(keras.Model):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        self.input_block = create_in_block_out_value(
            config.input_value_block,
            config.input_image_block,
            config.observation_space,
        )

        self.hidden_block = config.hidden_block.create_block_tf(config.action_space.n)

        # build
        self(np.zeros(self.input_block.create_batch_shape((1,)), config.dtype))

        self.loss_func = keras.losses.Huber()

    def call(self, x, training=False):
        x = self.input_block(x, training=training)
        x = self.hidden_block(x, training=training)
        return x

    @tf.function
    def compute_train_loss(self, state, onehot_action, target_q):
        q = self(state, training=True)
        q = tf.reduce_sum(q * onehot_action, axis=1)
        loss = self.loss_func(target_q, q)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss


class Parameter(RLParameter[Config]):
    def __init__(self, *args):
        super().__init__(*args)

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
    def __init__(self, *args):
        super().__init__(*args)

        self.lr_sch = SchedulerConfig.create_scheduler(self.config.lr)
        self.opt_q = keras.optimizers.Adam(self.lr_sch.get_rate())
        self.opt_snd_target = keras.optimizers.Adam(self.lr_sch.get_rate())
        self.opt_snd_predictor = keras.optimizers.Adam(self.lr_sch.get_rate())

        self.sync_count = 0

    def train(self) -> None:
        self._train_snd()
        self._train_q()

    def _train_snd(self):
        if self.memory.memory_snd.length() < self.config.memory_warmup_size:
            return

        # 対照学習
        # (s1, s1) -> tau=0
        # (s1, s2) -> tau=1
        batch_half = int(self.config.batch_size / 2)
        state1 = self.memory.sample_snd(self.config.batch_size)
        state2 = self.memory.sample_snd(batch_half)
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

    def _train_q(self):
        if self.memory.memory_q.length() < self.config.memory_warmup_size:
            return
        batchs = self.memory.sample_q()
        state, n_state, action, reward, undone = zip(*batchs)
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


class Worker(RLWorker[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)

    def on_reset(self, worker):
        self.memory.add("snd", worker.state)

    def policy(self, worker) -> int:
        invalid_actions = worker.get_invalid_actions()

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
        r_int = self._calc_intrinsic_reward(worker.state)
        reward = r_ext + self.config.int_reward_scale * r_int

        batch = [
            worker.prev_state,
            worker.state,
            funcs.one_hot(worker.action, self.config.action_space.n),
            reward,
            int(not worker.terminated),
        ]
        self.memory.add("q", batch)
        self.memory.add("snd", worker.state)

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

        funcs.render_discrete_action(int(maxa), self.config.action_space, worker.env, _render_sub)
