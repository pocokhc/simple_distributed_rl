import logging
import random
from typing import Any, List

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.algorithms.base_dqn import RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.trainer import RLTrainer
from srl.rl import functions as funcs
from srl.rl.functions import inverse_rescaling, rescaling
from srl.rl.memories.priority_replay_buffer import RLPriorityReplayBuffer
from srl.rl.tf.blocks.alphazero_image_block import AlphaZeroImageBlock
from srl.rl.tf.model import KerasModelAddedSummary

from .config import Config

kl = keras.layers
logger = logging.getLogger(__name__)


class Memory(RLPriorityReplayBuffer):
    def setup(self) -> None:
        super().setup()
        self.q_min = float("inf")
        self.q_max = float("-inf")
        self.register_worker_func(self.add_q, lambda x1, x2: (x1, x2))
        self.register_trainer_recv_func(self.get_q)

    def add_q(self, q_min, q_max, serialized: bool = False):
        self.q_min = min(self.q_min, q_min)
        self.q_max = max(self.q_max, q_max)

    def get_q(self):
        return self.q_min, self.q_max


class RepresentationNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config):
        super().__init__()

        self.in_block = config.input_image_block.create_tf_block(
            config.observation_space,
            out_flatten=False,
        )

        # build & 出力shapeを取得
        hidden_state = self(np.zeros(shape=(1,) + config.observation_space.shape, dtype=config.dtype))
        self.hidden_state_shape = hidden_state.shape[1:]

    def call(self, state, training=False):
        x = self.in_block(state, training=training)

        # 隠れ状態はアクションとスケールを合わせるため0-1で正規化(一応batch毎)
        batch, h, w, d = x.shape
        if batch is None:
            return x
        s_min = tf.reduce_min(tf.reshape(x, (batch, -1)), axis=1, keepdims=True)
        s_max = tf.reduce_max(tf.reshape(x, (batch, -1)), axis=1, keepdims=True)
        s_min = s_min * tf.ones((batch, h * w * d), dtype=tf.float32)
        s_max = s_max * tf.ones((batch, h * w * d), dtype=tf.float32)
        s_min = tf.reshape(s_min, (batch, h, w, d))
        s_max = tf.reshape(s_max, (batch, h, w, d))
        epsilon = 1e-4  # div0 回避
        x = (x - s_min + epsilon) / tf.maximum((s_max - s_min), 2 * epsilon)

        return x


class DynamicsNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, as_shape):
        super().__init__()
        self.tf_dtype = config.get_dtype("tf")
        self.reward_range = config.reward_range
        self.reward_range_num = config.reward_range_num
        self.c_size = config.codebook_size
        h, w, ch = as_shape

        # --- hidden_state
        self.image_block = AlphaZeroImageBlock(
            n_blocks=config.dynamics_blocks,
            filters=ch,
            # l2=config.weight_decay, TODO
        )

        # reward
        self.reward_layers = [
            kl.Conv2D(
                1,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Flatten(),
            kl.Dense(
                self.reward_range_num,
                activation="softmax",
                kernel_initializer="truncated_normal",
                bias_initializer="truncated_normal",
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
        ]

        # build
        self(np.zeros((1, h, w, ch + self.c_size)))

    def call(self, in_state, training=False):
        # hidden state
        x = self.image_block(in_state, training=training)

        # reward
        reward_category = in_state
        for layer in self.reward_layers:
            reward_category = layer(reward_category, training=training)

        return x, reward_category

    def predict(self, as_state, code_state, training=False):
        batch_size, h, w, _ = as_state.shape

        code_image = tf.repeat(code_state, h * w, axis=1)  # (b, c_size)->(b, c_size * h * w)
        code_image = tf.reshape(code_image, (batch_size, self.c_size, h, w))  # (b, c_size * h * w)->(b, c_size, h, w)
        code_image = tf.transpose(code_image, perm=[0, 2, 3, 1])  # (b, c_size, h, w)->(b, h, w, c_size)

        in_state = tf.concat([as_state, code_image], axis=3)
        x, reward_category = self.call(in_state, training=training)

        # 隠れ状態はアクションとスケールを合わせるため0-1で正規化(一応batch毎)
        batch, h, w, d = x.shape
        s_min = tf.reduce_min(tf.reshape(x, (batch, -1)), axis=1, keepdims=True)
        s_max = tf.reduce_max(tf.reshape(x, (batch, -1)), axis=1, keepdims=True)
        s_min = s_min * tf.ones((batch, h * w * d), dtype=tf.float32)
        s_max = s_max * tf.ones((batch, h * w * d), dtype=tf.float32)
        s_min = tf.reshape(s_min, (batch, h, w, d))
        s_max = tf.reshape(s_max, (batch, h, w, d))
        epsilon = 1e-4  # div0 回避
        x = (x - s_min + epsilon) / tf.maximum((s_max - s_min), 2 * epsilon)

        if training:
            return x, reward_category

        # reward
        reward = funcs.twohot_decode(
            reward_category.numpy(),
            self.reward_range_num,
            self.reward_range[0],
            self.reward_range[1],
        )
        return x, reward


class PredictionNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, s_state_shape):
        super().__init__()
        self.value_range = config.value_range
        self.value_range_num = config.value_range_num

        # --- policy
        self.policy_layers = [
            kl.Conv2D(
                2,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Flatten(),
            kl.Dense(
                config.action_space.n,
                activation="softmax",
                kernel_initializer="truncated_normal",
                bias_initializer="truncated_normal",
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
        ]

        # --- value
        self.value_layers = [
            kl.Conv2D(
                1,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Flatten(),
            kl.Dense(
                self.value_range_num,
                activation="softmax",
                kernel_initializer="truncated_normal",
                bias_initializer="truncated_normal",
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
        ]

        # build
        self(np.zeros((1,) + s_state_shape))

    def call(self, state, training=False):
        policy = state
        for layer in self.policy_layers:
            policy = layer(policy, training=training)

        value = state
        for layer in self.value_layers:
            value = layer(value, training=training)

        return policy, value

    def predict(self, state):
        p, v_category = self(state)
        p = p.numpy()
        v = funcs.twohot_decode(
            v_category.numpy(),
            self.value_range_num,
            self.value_range[0],
            self.value_range[1],
        )
        return p, v


class AfterstateDynamicsNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, hidden_shape):
        super().__init__()
        self.action_num = config.action_space.n
        h, w, ch = hidden_shape

        self.image_block = AlphaZeroImageBlock(
            n_blocks=config.dynamics_blocks,
            filters=ch,
            # l2=config.weight_decay_afterstate,  TODO
        )

        # build
        self(np.zeros((1, h, w, ch + self.action_num)))

    def call(self, in_state, training=False):
        return self.image_block(in_state, training=training)

    def predict(self, hidden_state, action, training=False):
        batch_size, h, w, _ = hidden_state.shape

        action_image = tf.one_hot(action, self.action_num)  # (batch, action)
        action_image = tf.repeat(action_image, repeats=h * w, axis=1)  # (batch, action * h * w)
        action_image = tf.reshape(action_image, (batch_size, self.action_num, h, w))  # (batch, action, h, w)
        action_image = tf.transpose(action_image, perm=[0, 2, 3, 1])  # (batch, h, w, action)

        # hidden_state + action_space
        in_state = tf.concat([hidden_state, action_image], axis=3)
        return self(in_state, training=training)


class AfterstatePredictionNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, as_shape):
        super().__init__()
        self.value_range = config.value_range
        self.value_range_num = config.value_range_num

        # --- code
        self.code_layers = [
            kl.Conv2D(
                2,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Flatten(),
            kl.Dense(
                config.codebook_size,
                activation="softmax",
                kernel_initializer="truncated_normal",
                bias_initializer="truncated_normal",
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
        ]

        # --- Q
        self.q_layers = [
            kl.Conv2D(
                1,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Flatten(),
            kl.Dense(
                self.value_range_num,
                activation="softmax",
                kernel_initializer="truncated_normal",
                bias_initializer="truncated_normal",
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
        ]
        # build
        self(np.zeros((1,) + as_shape))

    def call(self, state, training=False):
        code = state
        for layer in self.code_layers:
            code = layer(code, training=training)

        q = state
        for layer in self.q_layers:
            q = layer(q, training=training)

        return code, q

    def predict(self, state):
        code, q_category = self(state)
        q = funcs.twohot_decode(
            q_category.numpy(),
            self.value_range_num,
            self.value_range[0],
            self.value_range[1],
        )
        return code, q


class VQVAE(KerasModelAddedSummary):
    def __init__(self, config: Config):
        super().__init__()

        # --- codebook(one-hot vector)
        self.c_size = config.codebook_size
        self.codebook = np.identity(self.c_size, dtype=np.float32)[np.newaxis, ...]

        self.in_block = config.input_image_block.create_tf_block(
            config.observation_space,
            out_flatten=False,
        )

        self.out_layers = [
            kl.Conv2D(
                filters=2,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Flatten(),
            kl.Dense(
                config.codebook_size,
                activation="softmax",
                kernel_initializer="truncated_normal",
                bias_initializer="truncated_normal",
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
        ]

        # build
        self(np.zeros((1,) + config.observation_space.shape))

    def call(self, state, training=False):
        x = self.in_block(state, training=training)
        for layer in self.out_layers:
            x = layer(x, training=training)
        if state.shape[0] is None:
            return x
        return self.encode(x), x

    def encode_np(self, x):
        batch = x.shape[0]
        codebook = np.tile(self.codebook, (batch, 1, 1))  # [1, c, c]->[b, c, c]

        x = np.tile(x, (1, self.c_size))  # [b, c]->[b, c*c]
        x = np.reshape(x, (-1, self.c_size, self.c_size))  # [b, c*c]->[b, c, c]

        distance = np.sum((x - codebook) ** 2, axis=2)
        indices = np.argmin(distance, axis=1)
        onehot = np.identity(self.c_size, dtype=np.float32)[indices]
        onehot = np.tile(onehot, (1, self.c_size)).reshape((-1, self.c_size, self.c_size))  # [b, c, c]
        code = np.sum(onehot * codebook, axis=2)  # [b, c, c]->[b, c]
        return code

    def encode(self, x):
        batch_size = x.shape[0]

        # [1, c, c] → [b, c, c]
        codebook = tf.tile(self.codebook, [batch_size, 1, 1])

        # [b, c] → [b, c * c] → [b, c, c]
        x = tf.tile(x, [1, self.c_size])
        x = tf.reshape(x, [-1, self.c_size, self.c_size])

        # ユークリッド距離の計算 → [b, c]
        x = tf.reduce_sum(tf.square(x - codebook), axis=2)

        # 最小距離のインデックス → [b]
        x = tf.argmin(x, axis=1)

        # ワンホットエンコーディング → [b, c]
        x = tf.one_hot(x, depth=self.c_size, dtype=tf.float32)

        # [b, c] → [b, c * c] → [b, c, c]
        x = tf.tile(x, [1, self.c_size])
        x = tf.reshape(x, [-1, self.c_size, self.c_size])

        # [b, c, c] * [b, c, c] → sum over axis=2 → [b, c]
        code = tf.reduce_sum(x * codebook, axis=2)

        return code


class Parameter(RLParameter):
    def setup(self) -> None:
        self.representation_network = RepresentationNetwork(self.config)
        hidden_state_shape = self.representation_network.hidden_state_shape
        self.dynamics_network = DynamicsNetwork(self.config, hidden_state_shape)
        self.prediction_network = PredictionNetwork(self.config, hidden_state_shape)
        self.afterstate_dynamics_network = AfterstateDynamicsNetwork(self.config, hidden_state_shape)
        self.afterstate_prediction_network = AfterstatePredictionNetwork(self.config, hidden_state_shape)
        self.vq_vae = VQVAE(self.config)
        self.q_min = np.inf
        self.q_max = -np.inf

    def call_restore(self, data: Any, **kwargs) -> None:
        self.prediction_network.set_weights(data[0])
        self.dynamics_network.set_weights(data[1])
        self.representation_network.set_weights(data[2])
        self.afterstate_dynamics_network.set_weights(data[3])
        self.afterstate_prediction_network.set_weights(data[4])
        self.vq_vae.set_weights(data[5])
        self.q_min = data[6]
        self.q_max = data[7]

    def call_backup(self, **kwargs):
        return [
            self.prediction_network.get_weights(),
            self.dynamics_network.get_weights(),
            self.representation_network.get_weights(),
            self.afterstate_dynamics_network.get_weights(),
            self.afterstate_prediction_network.get_weights(),
            self.vq_vae.get_weights(),
            self.q_min,
            self.q_max,
        ]

    def summary(self, **kwargs):
        self.representation_network.summary(**kwargs)
        self.dynamics_network.summary(**kwargs)
        self.prediction_network.summary(**kwargs)
        self.afterstate_dynamics_network.summary(**kwargs)
        self.afterstate_prediction_network.summary(**kwargs)
        self.vq_vae.summary(**kwargs)


class Node:
    def __init__(self, prior: float, is_root):
        self.prior = prior
        self.is_root = is_root
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: List[Node] = []
        self.reward: float = 0.0
        self.score: float = 0.0
        self.v = 0.0
        self.s_state = None
        self.is_afterstate = False

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def expand(self, policy: List[float]) -> None:
        self.children = [Node(prior, is_root=False) for prior in policy]


class MCTS:
    def __init__(self, config: Config, parameter: Parameter) -> None:
        self.cfg = config
        self.parameter = parameter

    def simulation(self, root_state: np.ndarray, invalid_actions, training: bool):
        # --- root情報
        root_s_state = self.parameter.representation_network(root_state[np.newaxis, ...])
        p, v = self.parameter.prediction_network.predict(root_s_state)
        root = Node(prior=0.0, is_root=True)
        root.s_state = root_s_state
        root.is_afterstate = False
        root.v = v[0]
        root.expand(p[0])

        for _ in range(self.cfg.num_simulations):
            # --- 子ノードまで降りる
            node = root
            search_path = [node]
            while node.children:
                # select action
                action = self._select_action(node, invalid_actions, training)
                node = node.children[action]
                search_path.append(node)
                invalid_actions = []

            # --- expand
            parent_node = search_path[-2]
            if parent_node.is_afterstate:
                code, _ = self.parameter.afterstate_prediction_network.predict(parent_node.s_state)
                s_state, reward = self.parameter.dynamics_network.predict(parent_node.s_state, code)
                p, v = self.parameter.prediction_network.predict(s_state)
                node.is_afterstate = False
                node.reward = reward[0]
            else:
                s_state = self.parameter.afterstate_dynamics_network.predict(parent_node.s_state, [action])
                code, v = self.parameter.afterstate_prediction_network.predict(s_state)
                p = self.parameter.vq_vae.encode(code)
                node.is_afterstate = True
            node.s_state = s_state
            node.v = v[0]
            node.expand(p[0])

            # --- backup
            value = node.v
            for node in reversed(search_path):
                if not node.is_afterstate:
                    value = node.reward + self.cfg.discount * value
                node.value_sum += value
                node.visit_count += 1

                # 正規化用
                q = node.value
                self.parameter.q_min = min(self.parameter.q_min, q)
                self.parameter.q_max = max(self.parameter.q_max, q)

        return root

    def _select_action(self, node: Node, invalid_actions: list, training: bool):
        if node.is_root and training:
            dir_alpha = self.cfg.root_dirichlet_alpha
            if self.cfg.root_dirichlet_adaptive:
                dir_alpha = 1.0 / np.sqrt(self.cfg.action_space.n - len(invalid_actions))
            noises = np.random.dirichlet([dir_alpha] * self.cfg.action_space.n)
            e = self.cfg.root_dirichlet_fraction

        N = node.visit_count
        c = np.log((1 + N + self.cfg.c_base) / self.cfg.c_base) + self.cfg.c_init
        scores = np.zeros(self.cfg.action_space.n)
        for a, child in enumerate(node.children):
            n = child.visit_count
            p = child.prior
            q = child.value

            # rootはディリクレノイズを追加
            if node.is_root and training:
                p = (1 - e) * p + e * noises[a]

            # 過去観測したQ値で正規化(MinMax)
            if self.parameter.q_min < self.parameter.q_max:
                q = (q - self.parameter.q_min) / (self.parameter.q_max - self.parameter.q_min)

            node.score = q + c * p * (np.sqrt(N) / (1 + n))
            scores[a] = node.score

        scores[invalid_actions] = -np.inf
        action = int(np.random.choice(np.where(scores == np.max(scores))[0]))
        return action


def scale_gradient(tensor, scale):
    """muzeroより流用"""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def cross_entropy_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-6, y_pred)  # log(0)回避用
    loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    return loss


def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.np_dtype = self.config.get_dtype("np")
        self.opt_rep = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))
        self.opt_pre = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))
        self.opt_dyn = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))
        self.opt_after_dyn = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))
        self.opt_after_pre = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))
        self.opt_vq_vae = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return
        batches, weights, update_args = batches

        # (batch, steps, val) -> (steps, batch, val)
        states_list = []
        actions_list = []
        policies_list = []
        z_list = []
        rewards_list = []
        for i in range(self.config.unroll_steps + 1):
            states_list.append(np.asarray([b[i][0] for b in batches], dtype=self.np_dtype))
            actions_list.append(np.asarray([b[i][1] for b in batches], dtype=np.int64))
            policies_list.append(np.asarray([b[i][2] for b in batches], dtype=self.np_dtype))
            z_list.append(np.asarray([b[i][3] for b in batches], dtype=self.np_dtype))
            rewards_list.append(np.asarray([b[i][4] for b in batches], dtype=self.np_dtype))

        with tf.GradientTape() as tape:
            v_loss, p_loss, r_loss, chance_loss, q_loss, vae_loss = self._compute_train_loss(states_list, actions_list, rewards_list, policies_list, z_list)
            loss = v_loss + p_loss + r_loss + chance_loss + q_loss + self.config.commitment_cost * vae_loss
            loss = tf.reduce_mean(loss * weights)

            # 各ネットワークの正則化項を加える
            loss += tf.reduce_sum(self.parameter.representation_network.losses)
            loss += tf.reduce_sum(self.parameter.prediction_network.losses)
            loss += tf.reduce_sum(self.parameter.dynamics_network.losses)
            loss += tf.reduce_sum(self.parameter.afterstate_dynamics_network.losses)
            loss += tf.reduce_sum(self.parameter.afterstate_prediction_network.losses)
            loss += tf.reduce_sum(self.parameter.vq_vae.losses)

        variables = [
            self.parameter.representation_network.trainable_variables,
            self.parameter.dynamics_network.trainable_variables,
            self.parameter.prediction_network.trainable_variables,
            self.parameter.afterstate_dynamics_network.trainable_variables,
            self.parameter.afterstate_prediction_network.trainable_variables,
            self.parameter.vq_vae.trainable_variables,
        ]
        grads = tape.gradient(loss, variables)
        self.opt_rep.apply_gradients(zip(grads[0], variables[0]))
        self.opt_dyn.apply_gradients(zip(grads[1], variables[1]))
        self.opt_pre.apply_gradients(zip(grads[2], variables[2]))
        self.opt_after_dyn.apply_gradients(zip(grads[3], variables[3]))
        self.opt_after_pre.apply_gradients(zip(grads[4], variables[4]))
        self.opt_vq_vae.apply_gradients(zip(grads[5], variables[5]))

        self.train_count += 1
        self.info["loss"] = loss.numpy()
        self.info["v_loss"] = np.mean(v_loss)
        self.info["p_loss"] = np.mean(p_loss)
        self.info["r_loss"] = np.mean(r_loss)
        self.info["chance_loss"] = np.mean(chance_loss)
        self.info["q_loss"] = np.mean(q_loss)
        self.info["vae_loss"] = np.mean(vae_loss)

        # memory update
        priorities = np.abs(v_loss.numpy())
        self.memory.update(update_args, priorities, self.train_count)

        # --- 正規化用Qを保存(parameterはtrainerからしか保存されない)
        # (remote_memory -> trainer -> parameter)
        q = self.memory.get_q()
        if q is not None:
            self.parameter.q_min = min(self.parameter.q_min, q[0])
            self.parameter.q_max = max(self.parameter.q_max, q[1])

    @tf.function
    def _compute_train_loss(self, states_list, actions_list, rewards_list, policies_list, z_list):
        hidden_states = self.parameter.representation_network(states_list[0], training=True)

        # --- 1st step
        p_pred, v_pred = self.parameter.prediction_network(hidden_states, training=True)
        p_loss = cross_entropy_loss(policies_list[0], p_pred)
        v_loss = cross_entropy_loss(z_list[0], v_pred)
        r_loss = 0
        chance_loss = 0
        q_loss = 0
        vae_loss = 0

        # --- unroll steps
        gradient_scale = 1 / self.config.unroll_steps
        for t in range(self.config.unroll_steps):
            after_states = self.parameter.afterstate_dynamics_network.predict(hidden_states, actions_list[t], training=True)
            chance_pred, q_pred = self.parameter.afterstate_prediction_network(after_states, training=True)
            chance_code, chance_vae_pred = self.parameter.vq_vae(states_list[t + 1], training=True)

            chance_loss += scale_gradient(cross_entropy_loss(chance_code, chance_pred), gradient_scale)
            q_loss += scale_gradient(cross_entropy_loss(z_list[t], q_pred), gradient_scale)
            vae_loss += scale_gradient(mse_loss(chance_code, chance_vae_pred), gradient_scale)

            hidden_states, rewards_pred = self.parameter.dynamics_network.predict(after_states, chance_code, training=True)
            p_pred, v_pred = self.parameter.prediction_network(hidden_states)

            p_loss += scale_gradient(cross_entropy_loss(policies_list[t + 1], p_pred), gradient_scale)
            v_loss += scale_gradient(cross_entropy_loss(z_list[t + 1], v_pred), gradient_scale)
            r_loss += scale_gradient(cross_entropy_loss(rewards_list[t], rewards_pred), gradient_scale)

            hidden_states = scale_gradient(hidden_states, 0.5)

        v_loss /= self.config.unroll_steps + 1
        p_loss /= self.config.unroll_steps + 1
        r_loss /= self.config.unroll_steps
        chance_loss /= self.config.unroll_steps
        q_loss /= self.config.unroll_steps
        vae_loss /= self.config.unroll_steps
        return v_loss, p_loss, r_loss, chance_loss, q_loss, vae_loss


class Worker(RLWorker[Config, Parameter, Memory]):
    def on_setup(self, worker, context) -> None:
        self.np_dtype = self.config.get_dtype("np")
        self.env_player_num = worker.env.player_num
        self.policy_tau_sch = self.config.policy_tau_scheduler.create(self.config.policy_tau)
        self.mcts = MCTS(self.config, self.parameter)
        self.root = None

    def on_reset(self, worker):
        self.history = []

    def policy(self, worker) -> int:
        # --- シミュレーションしてpolicyを作成
        root = self.mcts.simulation(worker.state, worker.invalid_actions, self.training)
        self.step_v = root.v
        if self.rendering:
            self.root = root

        # --- 確率に比例したアクションを選択
        if self.training:
            policy_tau = self.policy_tau_sch.update(self.step_in_training).to_float()
        else:
            policy_tau = self.config.test_policy_tau

        action_select_count = np.array([n.visit_count for n in root.children])
        if policy_tau == 0:
            action = np.random.choice(np.where(action_select_count == action_select_count.max())[0])
        else:
            step_policy = np.maximum(action_select_count, 1e-8) ** (1 / policy_tau)
            step_policy /= step_policy.sum()
            action = funcs.random_choice_by_probs(step_policy)

        # 学習用のpolicyはtau=1
        self.step_policy = action_select_count / np.sum(action_select_count)

        self.info["policy_tau"] = policy_tau
        return int(action)

    def on_step(self, worker):
        if not self.training:
            return

        # 正規化用Qを保存できるように送信(memory -> trainer -> parameter)
        self.memory.add_q(self.parameter.q_min, self.parameter.q_max)
        self.info["q_min"] = self.parameter.q_min
        self.info["q_max"] = self.parameter.q_max

        reward = worker.reward
        if self.config.enable_rescale:
            reward = rescaling(reward)

        self.history.append(
            {
                "state": worker.state,
                "action": worker.action,
                "policy": self.step_policy,
                "reward": reward,
                "v": self.step_v,
            }
        )

        if worker.done:
            for _ in range(self.config.unroll_steps + 1):
                self.history.append(
                    {
                        "state": worker.state,
                        "action": random.randint(0, self.config.action_space.n - 1),
                        "policy": [1 / self.config.action_space.n for _ in range(self.config.action_space.n)],
                        "reward": 0,
                        "v": 0,
                    }
                )

            # --- calc discount reward
            reward = 0
            for h in reversed(self.history):
                reward = h["reward"] + self.config.discount * reward
                h["discount_reward"] = reward

                # twohot value
                h["twohot_z"] = funcs.twohot_encode(
                    h["discount_reward"],
                    self.config.value_range_num,
                    self.config.value_range[0],
                    self.config.value_range[1],
                    self.np_dtype,
                )

                # twohot reward
                h["twohot_reward"] = funcs.twohot_encode(
                    h["reward"],
                    self.config.reward_range_num,
                    self.config.reward_range[0],
                    self.config.reward_range[1],
                    self.np_dtype,
                )

            # --- add batch
            for idx in range(len(self.history) - self.config.unroll_steps - 1):
                batch = []
                priority = 0
                for i in range(self.config.unroll_steps + 1):
                    h = self.history[idx + i]
                    priority += abs(h["v"] - h["discount_reward"])
                    batch.append(
                        [
                            h["state"],
                            h["action"],
                            h["policy"],
                            h["twohot_z"],
                            h["twohot_reward"],
                        ]
                    )
                priority /= self.config.unroll_steps + 1
                self.memory.add(batch, priority)

    def render_terminal(self, worker, **kwargs) -> None:
        if self.root is None:
            return

        v = float(self.root.v)
        if self.config.enable_rescale:
            v = inverse_rescaling(v)
        print(f"V: {v:.5f}")

        s_state = self.parameter.representation_network(worker.state[np.newaxis, ...])
        children = self.root.children
        policy = self.step_policy

        def _render_sub(a: int) -> str:
            node = children[a]
            after_state = self.parameter.afterstate_dynamics_network.predict(s_state, [a])
            code, q = self.parameter.afterstate_prediction_network.predict(after_state)
            p = self.parameter.vq_vae.encode(code)
            _, reward = self.parameter.dynamics_network.predict(after_state, code)

            if self.config.enable_rescale:
                q = inverse_rescaling(q)
                reward = inverse_rescaling(reward)

            s = f"{policy[a] * 100:4.1f}%"
            s += f"({int(node.visit_count):3d})(N)"
            s += f" {node.value:5.3f}(Q)"
            s += f" {node.prior:6.3f}(P)"
            s += f" {node.score:6.3f}(PUCT)"
            s += f" {reward[0]:6.3f}(reward)"
            s += f", {q[0]:6.3f}(Q_pred)"
            s += f", {np.argmax(code[0]):d}(code)"
            s += f", {p[0][a]}(p)"
            return s

        worker.print_discrete_action_info(worker.action, _render_sub)
