import random
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.algorithms.base_dqn import RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.trainer import RLTrainer
from srl.rl.memories.replay_buffer import RLReplayBuffer
from srl.rl.tf.model import KerasModelAddedSummary

from .config import Config

kl = keras.layers


class Memory(RLReplayBuffer):
    pass


class QNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        self.in_block = config.input_block.create_tf_block(config)
        self.hidden_block = config.hidden_block.create_tf_block()
        self.out_layers = [
            kl.Dense(config.action_space.n * config.categorical_num_atoms),
            kl.Reshape((config.action_space.n, config.categorical_num_atoms)),
        ]

        # build
        self(np.zeros((1,) + config.observation_space.shape))

    def call(self, x, training=False):
        x = self.in_block(x, training=training)
        x = self.hidden_block(x, training=training)
        for h in self.out_layers:
            x = h(x)
        return x


class Parameter(RLParameter[Config]):
    def setup(self):
        self.Q = QNetwork(self.config)

    def call_restore(self, data: Any, **kwargs) -> None:
        self.Q.set_weights(data)

    def call_backup(self, **kwargs):
        return self.Q.get_weights()

    def summary(self, **kwargs):
        self.Q.summary(**kwargs)


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        lr = self.config.lr_scheduler.apply_tf_scheduler(self.config.lr)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

        self.n_atoms = self.config.categorical_num_atoms
        self.v_min = self.config.categorical_v_min
        self.v_max = self.config.categorical_v_max
        self.Z = np.linspace(self.v_min, self.v_max, self.n_atoms)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return

        states = []
        actions = []
        n_states = []
        rewards = []
        dones = []
        for b in batches:
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
        self.info["loss"] = loss.numpy()


class Worker(RLWorker[Config, Parameter, Memory]):
    def __init__(self, *args):
        super().__init__(*args)

        self.epsilon_sch = self.config.epsilon_scheduler.create(self.config.epsilon)
        self.Z = np.linspace(self.config.categorical_v_min, self.config.categorical_v_max, self.config.categorical_num_atoms)

    def policy(self, worker) -> int:
        state = worker.state
        invalid_actions = worker.invalid_actions

        if self.training:
            epsilon = self.epsilon_sch.update(self.step_in_training).to_float()
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
        self.info["epsilon"] = epsilon
        return int(action)

    def on_step(self, worker):
        if not self.training:
            return
        batch = {
            "state": worker.state,
            "next_state": worker.next_state,
            "action": self.action,
            "reward": worker.reward,
            "done": worker.terminated,
        }
        self.memory.add(batch)

    def render_terminal(self, worker, **kwargs) -> None:
        logits = self.parameter.Q(worker.state[np.newaxis, ...])
        probs = tf.nn.softmax(logits, axis=2)
        q_means = tf.reduce_sum(probs * self.Z, axis=2, keepdims=True)
        q = q_means[0].numpy().reshape(-1)
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        worker.print_discrete_action_info(int(maxa), _render_sub)
