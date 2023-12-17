import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import EnvObservationTypes, RLBaseTypes
from srl.base.rl.algorithms.discrete_action import DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.rl.registration import register
from srl.rl.functions.common import render_discrete_action
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, ExperienceReplayBufferConfig
from srl.rl.models.image_block import ImageBlockConfig
from srl.rl.models.mlp_block import MLPBlockConfig
from srl.rl.processors.image_processor import ImageProcessor
from srl.rl.schedulers.scheduler import SchedulerConfig

kl = keras.layers

"""
Categorical DQN（C51）
https://arxiv.org/abs/1707.06887

Other
    invalid_actions : TODO

"""


def create_input_layer(
    observation_shape: Tuple[int, ...],
    observation_type: EnvObservationTypes,
) -> Tuple[kl.Layer, kl.Layer, bool]:
    """状態の入力レイヤーを作成して返します

    Args:
        observation_shape (Tuple[int, ...]): 状態の入力shape
        observation_type (EnvObservationType): 状態が何かを表すEnvObservationType

    Returns:
        [
            in_layer  (kl.Layer): modelの入力に使うlayerを返します
            out_layer (kl.Layer): modelの続きに使うlayerを返します
            use_image_head (bool):
                Falseの時 out_layer は flatten、
                Trueの時 out_layer は CNN の形式で返ります。
        ]
    """

    # --- input
    in_layer = c = kl.Input(shape=observation_shape)
    err_msg = f"unknown observation_type: {observation_type}"

    # --- value head
    if (
        observation_type == EnvObservationTypes.DISCRETE
        or observation_type == EnvObservationTypes.CONTINUOUS
        or observation_type == EnvObservationTypes.UNKNOWN
    ):
        c = kl.Flatten()(c)
        return cast(kl.Layer, in_layer), cast(kl.Layer, c), False

    # --- image head
    if observation_type == EnvObservationTypes.GRAY_2ch:
        if len(observation_shape) == 2:
            # (w, h) -> (w, h, 1)
            c = kl.Reshape(observation_shape + (1,))(c)
        elif len(observation_shape) == 3:
            # (len, w, h) -> (w, h, len)
            c = kl.Permute((2, 3, 1))(c)
        else:
            raise ValueError(err_msg)

    elif observation_type == EnvObservationTypes.GRAY_3ch:
        assert observation_shape[-1] == 1
        if len(observation_shape) == 3:
            # (w, h, 1)
            pass
        elif len(observation_shape) == 4:
            # (len, w, h, 1) -> (len, w, h)
            # (len, w, h) -> (w, h, len)
            c = kl.Reshape(observation_shape[:3])(c)
            c = kl.Permute((2, 3, 1))(c)
        else:
            raise ValueError(err_msg)

    elif observation_type == EnvObservationTypes.COLOR:
        if len(observation_shape) == 3:
            # (w, h, ch)
            pass
        else:
            raise ValueError(err_msg)

    elif observation_type == EnvObservationTypes.SHAPE2:
        if len(observation_shape) == 2:
            # (w, h) -> (w, h, 1)
            c = kl.Reshape(observation_shape + (1,))(c)
        elif len(observation_shape) == 3:
            # (len, w, h) -> (w, h, len)
            c = kl.Permute((2, 3, 1))(c)
        else:
            raise ValueError(err_msg)

    elif observation_type == EnvObservationTypes.SHAPE3:
        if len(observation_shape) == 3:
            # (n, w, h) -> (w, h, n)
            c = kl.Permute((2, 3, 1))(c)
        else:
            raise ValueError(err_msg)

    else:
        raise ValueError(err_msg)

    return cast(kl.Layer, in_layer), cast(kl.Layer, c), True


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig, ExperienceReplayBufferConfig):
    test_epsilon: float = 0

    epsilon: float = 0.1  # type: ignore , type OK
    lr: float = 0.001  # type: ignore , type OK

    discount: float = 0.9

    # model
    image_block: ImageBlockConfig = field(init=False, default_factory=lambda: ImageBlockConfig())
    hidden_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())

    categorical_num_atoms: int = 51
    categorical_v_min: float = -10
    categorical_v_max: float = 10

    def __post_init__(self):
        super().__post_init__()

        self.epsilon: SchedulerConfig = SchedulerConfig(cast(float, self.epsilon))
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
    def base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    @property
    def base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.CONTINUOUS

    def get_use_framework(self) -> str:
        return "tensorflow"

    def getName(self) -> str:
        return "C51"

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
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        in_state, c, use_image_head = create_input_layer(
            self.config.observation_shape,
            self.config.env_observation_type,
        )
        if use_image_head:
            c = kl.Conv2D(32, (8, 8), strides=(4, 4), padding="same", activation="relu")(c)
            c = kl.Conv2D(64, (4, 4), strides=(4, 4), padding="same", activation="relu")(c)
            c = kl.Conv2D(64, (3, 3), strides=(4, 4), padding="same", activation="relu")(c)
            c = kl.Flatten()(c)

        # --- hidden block
        c = self.config.hidden_block.create_block_tf()(c)

        # --- out layer
        c = kl.Dense(self.config.action_num * self.config.categorical_num_atoms, activation="linear")(c)
        c = kl.Reshape((self.config.action_num, self.config.categorical_num_atoms))(c)
        self.Q = keras.Model(in_state, c)

    def call_restore(self, data: Any, **kwargs) -> None:
        self.Q.set_weights(data)

    def call_backup(self, **kwargs):
        return self.Q.get_weights()

    def summary(self, **kwargs):
        self.Q.summary(**kwargs)


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_sch = self.config.lr.create_schedulers()

        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch.get_rate())

        self.n_atoms = self.config.categorical_num_atoms
        self.v_min = self.config.categorical_v_min
        self.v_max = self.config.categorical_v_max
        self.Z = np.linspace(self.v_min, self.v_max, self.n_atoms)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

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
        actions = np.asarray(actions).reshape((-1, 1))

        #: a' = argmaxE[Z(s', a')]
        logits = self.parameter.Q(n_states)
        next_probs = tf.nn.softmax(logits, axis=2)
        q_means = tf.reduce_sum(next_probs * self.Z, axis=2, keepdims=True)
        next_actions = tf.argmax(q_means, axis=1)

        #: 選択されたaction軸だけ抽出する
        mask = np.ones((self.config.batch_size, self.config.action_num, self.n_atoms))
        onehot_mask = tf.one_hot(next_actions, self.config.action_num, axis=1)
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

        onehot_mask = tf.one_hot(actions, self.config.action_num, axis=1)
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
        self.train_info = {"loss": loss.numpy()}

        if self.lr_sch.update(self.train_count):
            lr = self.lr_sch.get_rate()
            self.optimizer.learning_rate = lr
            self.train_info["lr"] = lr


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.epsilon_sch = self.config.epsilon.create_schedulers()

        self.Z = np.linspace(
            self.config.categorical_v_min, self.config.categorical_v_max, self.config.categorical_num_atoms
        )

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        self.state = state
        self.invalid_actions = invalid_actions

        return {}

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        self.state = state
        self.invalid_actions = invalid_actions

        if self.training:
            epsilon = self.epsilon_sch.get_and_update_rate(self.total_step)
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            # epsilonより低いならランダム
            action = np.random.choice([a for a in range(self.config.action_num) if a not in invalid_actions])
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

    def call_on_step(
        self,
        next_state: Any,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ) -> Dict:
        if not self.training:
            return {}

        batch = {
            "state": self.state,
            "next_state": next_state,
            "action": self.action,
            "reward": reward,
            "done": done,
        }
        self.memory.add(batch)
        return {}

    def render_terminal(self, worker, **kwargs) -> None:
        logits = self.parameter.Q(self.state[np.newaxis, ...])
        probs = tf.nn.softmax(logits, axis=2)
        q_means = tf.reduce_sum(probs * self.Z, axis=2, keepdims=True)
        q = q_means[0].numpy().reshape(-1)
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        render_discrete_action(maxa, worker.env, self.config, _render_sub)
