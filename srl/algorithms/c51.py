import random
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
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.rl.functions import helper
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, RLConfigComponentExperienceReplayBuffer
from srl.rl.models.config.framework_config import RLConfigComponentFramework
from srl.rl.models.config.mlp_block import MLPBlockConfig
from srl.rl.models.tf.blocks.input_block import create_in_block_out_value
from srl.rl.schedulers.scheduler import SchedulerConfig

kl = keras.layers

"""
Categorical DQN（C51）
https://arxiv.org/abs/1707.06887

Other
    invalid_actions : TODO

"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(
    RLConfig[DiscreteSpace, BoxSpace],
    RLConfigComponentExperienceReplayBuffer,
    RLConfigComponentFramework,
):
    test_epsilon: float = 0

    epsilon: Union[float, SchedulerConfig] = 0.1
    lr: Union[float, SchedulerConfig] = 0.001

    discount: float = 0.9

    hidden_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())

    categorical_num_atoms: int = 51
    categorical_v_min: float = -10
    categorical_v_max: float = 10

    def __post_init__(self):
        super().__post_init__()

    def get_processors(self) -> List[Optional[ObservationProcessor]]:
        return [self.input_image_block.get_processor()]

    def get_base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.CONTINUOUS

    def get_framework(self) -> str:
        return "tensorflow"

    def get_name(self) -> str:
        return "C51"

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
# Parameter
# ------------------------------------------------------
class QNetwork(keras.Model):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        self.input_block = create_in_block_out_value(
            config.input_value_block,
            config.input_image_block,
            config.observation_space,
        )

        self.hidden_block = config.hidden_block.create_block_tf()

        self.out_layers = [
            kl.Dense(config.action_space.n * config.categorical_num_atoms),
            kl.Reshape((config.action_space.n, config.categorical_num_atoms)),
        ]

        # build
        self.build((None,) + config.observation_space.shape)

    def call(self, x, training=False):
        x = self.input_block(x, training=training)
        x = self.hidden_block(x, training=training)
        for h in self.out_layers:
            x = h(x)
        return x


class Parameter(RLParameter[Config]):
    def __init__(self, *args):
        super().__init__(*args)

        self.Q = QNetwork(self.config)

    def call_restore(self, data: Any, **kwargs) -> None:
        self.Q.set_weights(data)

    def call_backup(self, **kwargs):
        return self.Q.get_weights()

    def summary(self, **kwargs):
        self.Q.summary(**kwargs)


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)

        self.lr_sch = SchedulerConfig.create_scheduler(self.config.lr)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch.get_rate())

        self.n_atoms = self.config.categorical_num_atoms
        self.v_min = self.config.categorical_v_min
        self.v_max = self.config.categorical_v_max
        self.Z = np.linspace(self.v_min, self.v_max, self.n_atoms)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

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
        actions = np.asarray(actions).reshape((-1, 1))

        #: a' = argmaxE[Z(s', a')]
        logits = self.parameter.Q(n_states)
        next_probs = tf.nn.softmax(logits, axis=2)
        q_means = tf.reduce_sum(next_probs * self.Z, axis=2, keepdims=True)
        next_actions = tf.argmax(q_means, axis=1)

        #: 選択されたaction軸だけ抽出する
        mask = np.ones((self.config.batch_size, self.config.action_space.n, self.n_atoms))
        onehot_mask = tf.one_hot(next_actions, self.config.action_space.n, axis=1)
        onehot_mask = onehot_mask * mask
        next_dists = tf.reduce_sum(next_probs * onehot_mask, axis=1).numpy()

        #: 分布版ベルマンオペレータの適用
        rewards = np.tile(np.reshape(rewards, (-1, 1)), (1, self.n_atoms))
        dones = np.tile(np.reshape(dones, (-1, 1)), (1, self.n_atoms))
        Z = np.tile(self.Z, (self.config.batch_size, 1))
        TZ = rewards + (1 - dones) * self.config.discount * Z

        # 設定区間を超えないようにクリップ
        TZ = np.minimum(self.v_max, np.maximum(self.v_min, TZ))

        # 再割り当て
        target_dists = np.zeros((self.config.batch_size, self.config.categorical_num_atoms))
        bj = (TZ - self.v_min) / self.delta_z
        ratios, indices = np.modf(bj)
        for i in range(self.config.batch_size):
            for j in range(self.n_atoms):
                idx = int(indices[i][j])
                ratio = ratios[i][j]
                target_dists[i][idx] += next_dists[i][j] * (1 - ratio)
                if ratio != 0:
                    target_dists[i][idx + 1] += next_dists[i][j] * ratio

        onehot_mask = tf.one_hot(actions, self.config.action_space.n, axis=1)
        onehot_mask = onehot_mask * mask

        with tf.GradientTape() as tape:
            logits = self.parameter.Q(states)
            probs = tf.nn.softmax(logits, axis=2)

            dists = tf.reduce_sum(probs * onehot_mask, axis=1)
            dists = tf.clip_by_value(dists, 1e-6, 1.0)

            #: categorical cross entropy
            loss = tf.reduce_sum(-1 * target_dists * tf.math.log(dists), axis=1, keepdims=True)  # type:ignore
            loss = tf.reduce_mean(loss)
            loss += tf.reduce_sum(self.parameter.Q.losses)  # 正則化のLoss

        grads = tape.gradient(loss, self.parameter.Q.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.Q.trainable_variables))

        self.train_count += 1
        self.info = {"loss": loss.numpy()}

        if self.lr_sch.update(self.train_count):
            lr = self.lr_sch.get_rate()
            self.optimizer.learning_rate = lr
            self.info["lr"] = lr


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)

        self.epsilon_sch = SchedulerConfig.create_scheduler(self.config.epsilon)

        self.Z = np.linspace(
            self.config.categorical_v_min, self.config.categorical_v_max, self.config.categorical_num_atoms
        )

    def on_reset(self, worker) -> InfoType:
        return {}

    def policy(self, worker) -> Tuple[int, InfoType]:
        state = worker.state
        invalid_actions = worker.get_invalid_actions()

        if self.training:
            epsilon = self.epsilon_sch.get_and_update_rate(self.total_step)
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            # epsilonより低いならランダム
            action = np.random.choice([a for a in range(self.config.action_space.n) if a not in invalid_actions])
        else:
            logits = self.parameter.Q(np.asarray([state]))
            probs = tf.nn.softmax(logits, axis=2)
            q_means = tf.reduce_sum(probs * self.Z, axis=2, keepdims=True).numpy()[0]
            q_means = q_means.reshape(-1)

            # valid actions以外は -inf にする
            q = np.array([(-np.inf if a in invalid_actions else v) for a, v in enumerate(q_means)])

            # 最大値を選ぶ（複数あればランダム）
            action = np.random.choice(np.where(q == q.max())[0])

        self.action = action
        return int(action), {"epsilon": epsilon}

    def on_step(self, worker) -> InfoType:
        if not self.training:
            return {}

        batch = {
            "state": worker.prev_state,
            "next_state": worker.state,
            "action": self.action,
            "reward": worker.reward,
            "done": worker.terminated,
        }
        self.memory.add(batch)
        return {}

    def render_terminal(self, worker, **kwargs) -> None:
        logits = self.parameter.Q(worker.prev_state[np.newaxis, ...])
        probs = tf.nn.softmax(logits, axis=2)
        q_means = tf.reduce_sum(probs * self.Z, axis=2, keepdims=True)
        q = q_means[0].numpy().reshape(-1)
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        helper.render_discrete_action(int(maxa), self.config.action_space.n, worker.env, _render_sub)
