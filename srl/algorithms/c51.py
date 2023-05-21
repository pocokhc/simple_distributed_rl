import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import EnvObservationTypes, RLObservationTypes
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.model import IImageBlockConfig, IMLPBlockConfig
from srl.base.rl.processor import Processor
from srl.base.rl.processors.image_processor import ImageProcessor
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import ExperienceReplayBuffer
from srl.rl.functions.common import render_discrete_action
from srl.rl.models.dqn.dqn_image_block_config import DQNImageBlockConfig
from srl.rl.models.mlp.mlp_block_config import MLPBlockConfig
from srl.rl.models.tf.input_block import InputBlock

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
class Config(DiscreteActionConfig):
    epsilon: float = 0.1
    test_epsilon: float = 0
    discount: float = 0.9
    lr: float = 0.001
    batch_size: int = 16
    memory_warmup_size: int = 1000
    capacity: int = 100_000

    # model
    image_block_config: IImageBlockConfig = field(default_factory=lambda: DQNImageBlockConfig())
    hidden_block: IMLPBlockConfig = field(default_factory=lambda: MLPBlockConfig())

    categorical_num_atoms: int = 51
    categorical_v_min: float = -10
    categorical_v_max: float = 10

    def set_processor(self) -> List[Processor]:
        return [
            ImageProcessor(
                image_type=EnvObservationTypes.GRAY_2ch,
                resize=(84, 84),
                enable_norm=True,
            )
        ]

    @property
    def observation_type(self) -> RLObservationTypes:
        return RLObservationTypes.CONTINUOUS

    def getName(self) -> str:
        return "C51"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.memory_warmup_size < self.capacity
        assert self.batch_size < self.memory_warmup_size


register(
    Config(),
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
class _QNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        # input
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)

        # image
        if self.in_block.use_image_layer:
            self.image_block = config.image_block_config.create_block_tf()
            self.image_flatten = kl.Flatten()

        # hidden
        self.hidden_block = config.hidden_block.create_block_tf()

        # out
        self.out_layers = [
            kl.Dense(
                self.config.action_num * self.config.categorical_num_atoms,
                activation="linear",
            ),
            kl.Reshape((self.config.action_num, self.config.categorical_num_atoms)),
        ]

        # build
        self.build((None,) + config.observation_shape)

    def call(self, x, training=False):
        x = self.in_block(x, training=training)
        if self.in_block.use_image_layer:
            x = self.image_block(x, training=training)
            x = self.image_flatten(x)
        x = self.hidden_block(x, training=training)
        for layer in self.out_layers:
            x = layer(x, training=training)
        return x

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def summary(self, name="", **kwargs):
        if hasattr(self.in_block, "init_model_graph"):
            self.in_block.init_model_graph()
        if self.in_block.use_image_layer and hasattr(self.image_block, "init_model_graph"):
            self.image_block.init_model_graph()

        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x), name=name)
        model.summary(**kwargs)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

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
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        self.optimizer = keras.optimizers.Adam(learning_rate=self.config.lr)

        self.train_count = 0

        self.n_atoms = self.config.categorical_num_atoms
        self.v_min = self.config.categorical_v_min
        self.v_max = self.config.categorical_v_max
        self.Z = np.linspace(self.v_min, self.v_max, self.n_atoms)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

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
        return {"loss": loss.numpy()}


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        self.Z = np.linspace(
            self.config.categorical_v_min, self.config.categorical_v_max, self.config.categorical_num_atoms
        )

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        self.state = state
        self.invalid_actions = invalid_actions

        if self.training:
            self.epsilon = self.config.epsilon
        else:
            self.epsilon = self.config.test_epsilon

        return {}

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        self.state = state
        self.invalid_actions = invalid_actions

        if random.random() < self.epsilon:
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
        return action, {}

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
        self.remote_memory.add(batch)
        return {}

    def render_terminal(self, env, worker, **kwargs) -> None:
        logits = self.parameter.Q(self.state[np.newaxis, ...])
        probs = tf.nn.softmax(logits, axis=2)
        q_means = tf.reduce_sum(probs * self.Z, axis=2, keepdims=True)
        q = q_means[0].numpy().reshape(-1)
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        render_discrete_action(self.invalid_actions, maxa, env, _render_sub)
