import random
from dataclasses import dataclass
from typing import Any, Dict, List, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kl
from srl.base.define import RLObservationType
from srl.base.env.base import EnvRun
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import ExperienceReplayBuffer
from srl.rl.functions.common import render_discrete_action
from srl.rl.models.dqn_image_block import DQNImageBlock
from srl.rl.models.input_layer import create_input_layer
from srl.rl.models.mlp_block import MLPBlock

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
class Config(DiscreteActionConfig):

    epsilon: float = 0.1
    test_epsilon: float = 0
    gamma: float = 0.9
    lr: float = 0.001
    batch_size: int = 16
    memory_warmup_size: int = 1000
    capacity: int = 100_000

    # model
    cnn_block: kl.Layer = DQNImageBlock
    cnn_block_kwargs: dict = None
    hidden_block: kl.Layer = MLPBlock
    hidden_block_kwargs: dict = None

    categorical_num_atoms: int = 51
    categorical_v_min: float = -10
    categorical_v_max: float = 10

    def __post_init__(self):
        super().__init__()
        if self.cnn_block_kwargs is None:
            self.cnn_block_kwargs = {}
        if self.hidden_block_kwargs is None:
            self.hidden_block_kwargs = {}

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.CONTINUOUS

    @staticmethod
    def getName() -> str:
        return "C51"

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
            c = self.config.cnn_block(**self.config.cnn_block_kwargs)(c)
            c = kl.Flatten()(c)

        # --- hidden block
        c = self.config.hidden_block(**self.config.hidden_block_kwargs)(c)

        # --- out layer
        c = kl.Dense(self.config.action_num * self.config.categorical_num_atoms, activation="linear")(c)
        c = kl.Reshape((self.config.action_num, self.config.categorical_num_atoms))(c)
        self.Q = keras.Model(in_state, c)

    def restore(self, data: Any) -> None:
        self.Q.set_weights(data)

    def backup(self):
        return self.Q.get_weights()

    def summary(self, **kwargs):
        self.Q.summary()


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
        TZ = rewards + (1 - dones) * self.config.gamma * Z

        # 設定区間を超えないようにクリップ
        TZ = np.minimum(self.v_max, np.maximum(self.v_min, TZ))

        # 再割り当て
        target_dists = np.zeros((self.config.batch_size, self.config.categorical_num_atoms))
        bj = (TZ - self.v_min) / self.delta_z
        ratios, indexes = np.modf(bj)
        for i in range(self.config.batch_size):
            for j in range(self.n_atoms):
                idx = int(indexes[i][j])
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
            loss = tf.reduce_sum(-1 * target_dists * tf.math.log(dists), axis=1, keepdims=True)
            loss = tf.reduce_mean(loss)

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

        self.v_min = -10
        self.v_max = 10
        self.Z = np.linspace(self.v_min, self.v_max, self.config.categorical_num_atoms)

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> None:
        self.state = state
        self.invalid_actions = invalid_actions

        if self.training:
            self.epsilon = self.config.epsilon
        else:
            self.epsilon = self.config.test_epsilon

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> int:
        self.state = state
        self.invalid_actions = invalid_actions

        if random.random() < self.epsilon:
            # epsilonより低いならランダム
            action = random.choice([a for a in range(self.config.action_num) if a not in invalid_actions])
        else:
            logits = self.parameter.Q(np.asarray([state]))
            probs = tf.nn.softmax(logits, axis=2)
            q_means = tf.reduce_sum(probs * self.Z, axis=2, keepdims=True).numpy()[0]
            q_means = q_means.reshape(-1)

            # valid actions以外は -inf にする
            q = np.array([(-np.inf if a in invalid_actions else v) for a, v in enumerate(q_means)])

            # 最大値を選ぶ（複数あればランダム）
            action = random.choice(np.where(q == q.max())[0])

        self.action = action
        return action

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

    def call_render(self, env: EnvRun) -> None:
        logits = self.parameter.Q(self.state[np.newaxis, ...])
        probs = tf.nn.softmax(logits, axis=2)
        q_means = tf.reduce_sum(probs * self.Z, axis=2, keepdims=True)
        q = q_means[0].numpy().reshape(-1)
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        render_discrete_action(self.invalid_actions, maxa, env, _render_sub)
