import logging
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.algorithms.base_dqn import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import RLProcessor
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.base.spaces.space import SpaceBase
from srl.rl import functions as funcs
from srl.rl.functions import inverse_rescaling, rescaling
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig, RLPriorityReplayBuffer
from srl.rl.models.config.input_image_block import InputImageBlockConfig
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig
from srl.rl.schedulers.scheduler import SchedulerConfig
from srl.rl.tf.blocks.alphazero_image_block import AlphaZeroImageBlock
from srl.rl.tf.model import KerasModelAddedSummary

kl = keras.layers
logger = logging.getLogger(__name__)

"""
Paper
https://arxiv.org/abs/1911.08265

Ref
https://arxiv.org/src/1911.08265v2/anc/pseudocode.py
"""


@dataclass
class Config(RLConfig):
    #: シミュレーション回数
    num_simulations: int = 50
    #: 割引率
    discount: float = 0.99

    #: Batch size
    batch_size: int = 32
    #: <:ref:`PriorityReplayBufferConfig`>
    memory: PriorityReplayBufferConfig = field(default_factory=lambda: PriorityReplayBufferConfig())

    #: <:ref:`InputImageBlockConfig`>
    input_image_block: InputImageBlockConfig = field(default_factory=lambda: InputImageBlockConfig().set_alphazero_block())
    #: Learning rate
    lr: float = 0.001
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: (LRSchedulerConfig().set_step(100_000, 0.0001)))
    #: カテゴリ化する範囲
    reward_range: tuple = (-10, 10)
    reward_range_num: int = 100
    #: カテゴリ化する範囲
    value_range: tuple = (-10, 10)
    value_range_num: int = 100

    test_policy_tau: float = 0.1
    # policyの温度パラメータのリスト
    policy_tau: Optional[float] = None
    #: <:ref:`SchedulerConfig`>
    policy_tau_scheduler: SchedulerConfig = field(
        default_factory=lambda: (
            SchedulerConfig(default_scheduler=True)  #
            .add_constant(1.0, 50_000)
            .add_constant(0.5, 25_000)
            .add_constant(0.25)
        )
    )
    # td_steps: int = 5  # MC法でエピソード最後まで展開しているので未使用
    #: unroll_steps
    unroll_steps: int = 3

    #: Root prior exploration noise.
    root_dirichlet_alpha: float = 0.3
    #: Root prior exploration noise.
    root_exploration_fraction: float = 0.25

    #: PUCT
    c_base: float = 19652
    #: PUCT
    c_init: float = 1.25

    #: Dynamics networkのブロック数
    dynamics_blocks: int = 15
    #: reward dense units
    reward_dense_units: int = 0
    #: weight decay
    weight_decay: float = 0.0001

    #: rescale
    enable_rescale: bool = False
    #: reanalyze
    enable_reanalyze: bool = False

    def set_atari_config(self):
        self.num_simulations = 50
        self.batch_size = 1024
        self.memory.warmup_size = 10_000
        self.discount = 0.997
        self.lr = 0.05
        self.lr_scheduler.set_step(350_000, 0.005)
        self.reward_range = (-300, 300)
        self.reward_range_num = 601
        self.value_range = (-300, 300)
        self.value_range_num = 601
        # self.td_steps = 10
        self.unroll_steps = 5
        self.policy_tau_scheduler.clear()
        self.policy_tau_scheduler.add_constant(1.0, 500_000)
        self.policy_tau_scheduler.add_constant(0.5, 250_000)
        self.policy_tau_scheduler.add_constant(0.25)
        self.input_image_block.set_muzero_atari_block(filters=128)
        self.dynamics_blocks = 15
        self.weight_decay = 0.0001
        self.enable_rescale = True
        return self

    def set_board_game_config(self):
        self.num_simulations = 800
        self.batch_size = 2048
        self.memory.warmup_size = 10_000
        self.discount = 1.0
        self.lr = 0.05
        self.lr_scheduler.set_step(400_000, 0.005)
        self.reward_range = (-300, 300)
        self.reward_range_num = 601
        self.value_range = (-300, 300)
        self.value_range_num = 601
        # self.td_steps = 10
        self.unroll_steps = 10
        self.policy_tau_scheduler.clear()
        self.policy_tau_scheduler.add_constant(1.0, 500_000)
        self.policy_tau_scheduler.add_constant(0.5, 250_000)
        self.policy_tau_scheduler.add_constant(0.25)
        self.input_image_block.set_alphazero_block()
        self.dynamics_blocks = 15
        self.weight_decay = 0.0001
        self.enable_rescale = True
        return self

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if not prev_observation_space.is_image():
            raise ValueError(f"The input supports only image format. {prev_observation_space}")
        return self.input_image_block.get_processors()

    def get_name(self) -> str:
        return "MuZero"

    def get_framework(self) -> str:
        return "tensorflow"

    def validate_params(self) -> None:
        super().validate_params()
        if not (self.value_range[0] < self.value_range[1]):
            raise ValueError(f"assert {self.value_range[0]} < {self.value_range[1]}")
        if not (self.reward_range[0] < self.reward_range[1]):
            raise ValueError(f"assert {self.reward_range[0]} < {self.reward_range[1]}")
        if not (self.unroll_steps > 0):
            raise ValueError(f"assert {self.unroll_steps} > 0")
        if self.enable_reanalyze:
            raise ValueError("Not implemented")


register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


class Memory(RLPriorityReplayBuffer[Config]):
    def setup(self) -> None:
        super().setup()
        self.np_dtype = self.config.get_dtype("np")
        self.q_min = float("inf")
        self.q_max = float("-inf")
        self.register_worker_func(self.add_q, lambda x1, x2: (x1, x2))
        self.register_trainer_recv_func(self.get_q)

        self.register_trainer_send_func(self.update_parameter)
        self.register_trainer_recv_func(self.sample_batch)

        if self.config.enable_reanalyze:
            self.parameter = Parameter(self.config)
            self.mcts = MCTS(self.config, self.parameter)

    def add_q(self, q_min, q_max, serialized: bool = False):
        self.q_min = min(self.q_min, q_min)
        self.q_max = max(self.q_max, q_max)

    def get_q(self):
        return self.q_min, self.q_max

    def sample_batch(self):
        batches = self.sample()
        if batches is None:
            return None
        batches, weights, update_args = batches

        # (batch, steps, val) -> (batch, val)
        state = np.asarray([steps[0][0] for steps in batches], dtype=self.np_dtype)

        # (batch, steps, val) -> (steps, batch, val)
        actions_list = []
        policies_list = []
        z_list = []
        rewards_list = []
        for i in range(self.config.unroll_steps + 1):
            actions_list.append(np.asarray([b[i][1] for b in batches], dtype=np.int64))
            policies_list.append(np.asarray([b[i][2] for b in batches], dtype=self.np_dtype))
            z_list.append(np.asarray([b[i][3] for b in batches], dtype=self.np_dtype))
            rewards_list.append(np.asarray([b[i][4] for b in batches], dtype=self.np_dtype))

        if not self.config.enable_reanalyze:
            return state, actions_list, rewards_list, policies_list, z_list, weights, update_args

        # --- reanalyze
        # TODO: 実装するならここ
        s_state = self.parameter.representation_network(state)
        for i in range(self.config.unroll_steps + 1):
            s_state, p_rewards = self.parameter.dynamics_network.predict(s_state, actions_list[i], training=True)

        return state, actions_list, rewards_list, policies_list, z_list, weights, update_args

    def update_parameter(self, dat):
        self.parameter.restore(dat)


class RepresentationNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config):
        super().__init__()
        self.tf_dtype = config.get_dtype("tf")

        self.in_block = config.input_image_block.create_tf_block(
            config.observation_space,
            out_flatten=False,
        )

        # build & 出力shapeを取得
        s_state = self(np.zeros(shape=(1,) + config.observation_space.shape, dtype=config.dtype))
        self.s_state_shape = s_state.shape[1:]

    def call(self, state, training=False):
        x = self.in_block(state, training=training)

        # 隠れ状態はアクションとスケールを合わせるため0-1で正規化(一応batch毎)
        batch, h, w, d = x.shape
        if batch is None:
            return x
        s_min = tf.reduce_min(tf.reshape(x, (batch, -1)), axis=1, keepdims=True)
        s_max = tf.reduce_max(tf.reshape(x, (batch, -1)), axis=1, keepdims=True)
        s_min = s_min * tf.ones((batch, h * w * d), dtype=self.tf_dtype)
        s_max = s_max * tf.ones((batch, h * w * d), dtype=self.tf_dtype)
        s_min = tf.reshape(s_min, (batch, h, w, d))
        s_max = tf.reshape(s_max, (batch, h, w, d))
        epsilon = 1e-4  # div0 回避
        x = (x - s_min + epsilon) / tf.maximum((s_max - s_min), 2 * epsilon)

        return x


class DynamicsNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, input_shape):
        super().__init__()
        self.tf_dtype = config.get_dtype("tf")
        self.action_num = config.action_space.n
        self.reward_range = config.reward_range
        self.reward_range_num = config.reward_range_num

        h, w, ch = input_shape

        # --- hidden_state
        self.image_block = AlphaZeroImageBlock(
            n_blocks=config.dynamics_blocks,
            filters=ch,
            # l2=config.weight_decay, TODO
        )

        # --- reward
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
        ]
        if config.reward_dense_units > 0:
            self.reward_layers.append(
                kl.Dense(
                    config.reward_dense_units,
                    activation="swish",
                    kernel_regularizer=keras.regularizers.l2(config.weight_decay),
                )
            )
        self.reward_layers.append(
            kl.Dense(
                self.reward_range_num,
                activation="softmax",
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            )
        )

        # build
        self(np.zeros((1, h, w, ch + self.action_num)))

    def call(self, in_state, training=False):
        # hidden state
        x = self.image_block(in_state, training=training)

        # reward
        reward_category = in_state
        for layer in self.reward_layers:
            reward_category = layer(reward_category, training=training)

        return x, reward_category

    def predict(self, hidden_state, action, training=False):
        batch_size, h, w, _ = hidden_state.shape

        action_image = tf.one_hot(action, self.action_num, dtype=self.tf_dtype)  # (batch, action)
        action_image = tf.repeat(action_image, repeats=h * w, axis=1)  # (batch, action * h * w)
        action_image = tf.reshape(action_image, (batch_size, self.action_num, h, w))  # (batch, action, h, w)
        action_image = tf.transpose(action_image, perm=[0, 2, 3, 1])  # (batch, h, w, action)

        in_state = tf.concat([hidden_state, action_image], axis=3)
        x, reward_category = self(in_state, training=training)

        # 隠れ状態はアクションとスケールを合わせるため0-1で正規化(一応batch毎)
        batch, h, w, d = x.shape
        s_min = tf.reduce_min(tf.reshape(x, (batch, -1)), axis=1, keepdims=True)
        s_max = tf.reduce_max(tf.reshape(x, (batch, -1)), axis=1, keepdims=True)
        s_min = s_min * tf.ones((batch, h * w * d), dtype=self.tf_dtype)
        s_max = s_max * tf.ones((batch, h * w * d), dtype=self.tf_dtype)
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
    def __init__(self, config: Config, input_shape):
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
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
        ]

        # build
        self(np.zeros((1,) + input_shape))

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


class Parameter(RLParameter[Config]):
    def setup(self) -> None:
        self.representation_network = RepresentationNetwork(self.config)
        s_state_shape = self.representation_network.s_state_shape
        self.prediction_network = PredictionNetwork(self.config, s_state_shape)
        self.dynamics_network = DynamicsNetwork(self.config, s_state_shape)

        self.q_min = np.inf
        self.q_max = -np.inf

    def call_restore(self, data: Any, **kwargs) -> None:
        self.prediction_network.set_weights(data[0])
        self.dynamics_network.set_weights(data[1])
        self.representation_network.set_weights(data[2])
        self.q_min = data[3]
        self.q_max = data[4]

    def call_backup(self, **kwargs):
        return [
            self.prediction_network.get_weights(),
            self.dynamics_network.get_weights(),
            self.representation_network.get_weights(),
            self.q_min,
            self.q_max,
        ]

    def summary(self, **kwargs):
        self.representation_network.summary(**kwargs)
        self.dynamics_network.summary(**kwargs)
        self.prediction_network.summary(**kwargs)


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
            parent_node_state = search_path[-2].s_state
            s_state, reward = self.parameter.dynamics_network.predict(parent_node_state, [action])
            p, v = self.parameter.prediction_network.predict(s_state)
            node.s_state = s_state
            node.reward = reward[0]
            node.v = v[0]
            node.expand(p[0])

            # --- backup
            value = node.v
            for node in reversed(search_path):
                node.visit_count += 1
                value = node.reward + self.cfg.discount * value
                node.value_sum += value

                # 正規化用
                q = node.value
                self.parameter.q_min = min(self.parameter.q_min, q)
                self.parameter.q_max = max(self.parameter.q_max, q)

        return root

    def _select_action(self, node: Node, invalid_actions: list, training: bool):
        if node.is_root and training:
            noises = np.random.dirichlet([self.cfg.root_dirichlet_alpha] * self.cfg.action_space.n)
            e = self.cfg.root_exploration_fraction

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
    """
    Scales the gradient for the backward pass.
    論文の疑似コードより
    """
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def cross_entropy_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-6, y_pred)  # log(0)回避用
    loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1)
    return loss


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.np_dtype = self.config.get_dtype("np")
        self.tf_dtype = self.config.get_dtype("tf")
        self.opt_rep = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))
        self.opt_pre = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))
        self.opt_dyn = keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))

    def train(self) -> None:
        batches = self.memory.sample_batch()
        if batches is None:
            return
        state, actions_list, rewards_list, policies_list, z_list, weights, update_args = batches

        with tf.GradientTape() as tape:
            v_loss, p_loss, r_loss = self._compute_train_loss(state, actions_list, rewards_list, policies_list, z_list)
            loss = tf.reduce_mean((v_loss + p_loss + r_loss) * weights)

            # 正則化項
            loss += tf.reduce_sum(self.parameter.representation_network.losses)
            loss += tf.reduce_sum(self.parameter.prediction_network.losses)
            loss += tf.reduce_sum(self.parameter.dynamics_network.losses)

        variables = [
            self.parameter.representation_network.trainable_variables,
            self.parameter.prediction_network.trainable_variables,
            self.parameter.dynamics_network.trainable_variables,
        ]
        grads = tape.gradient(loss, variables)
        self.opt_rep.apply_gradients(zip(grads[0], variables[0]))
        self.opt_pre.apply_gradients(zip(grads[1], variables[1]))
        self.opt_dyn.apply_gradients(zip(grads[2], variables[2]))

        self.train_count += 1
        self.info["loss"] = loss.numpy()
        self.info["value_loss"] = np.mean(v_loss)
        self.info["policy_loss"] = np.mean(p_loss)
        self.info["reward_loss"] = np.mean(r_loss)

        # memory update
        priorities = np.abs(v_loss.numpy())
        self.memory.update(update_args, priorities, self.train_count)

        # --- 正規化用Qを保存(parameterはtrainerからしか保存されない)
        # (remote_memory -> trainer -> parameter)
        q = self.memory.get_q()
        if q is not None:
            self.parameter.q_min = min(self.parameter.q_min, q[0])
            self.parameter.q_max = max(self.parameter.q_max, q[1])

        # parameter update
        if self.config.enable_reanalyze:
            self.memory.update_parameter(self.parameter.backup())

    @tf.function
    def _compute_train_loss(self, state, actions_list: list, rewards_list: list, policies_list: list, z_list: list):
        s_state = self.parameter.representation_network(state, training=True)

        # --- 1st step
        p_pred, v_pred = self.parameter.prediction_network(s_state, training=True)
        p_loss = scale_gradient(cross_entropy_loss(policies_list[0], p_pred), 1.0)
        v_loss = scale_gradient(cross_entropy_loss(z_list[0], v_pred), 1.0)
        r_loss = 0

        # --- unroll steps
        gradient_scale = 1 / self.config.unroll_steps
        for t in range(self.config.unroll_steps):
            s_state, rewards_pred = self.parameter.dynamics_network.predict(s_state, actions_list[t], training=True)
            p_pred, v_pred = self.parameter.prediction_network(s_state, training=True)

            p_loss += scale_gradient(cross_entropy_loss(policies_list[t + 1], p_pred), gradient_scale)
            v_loss += scale_gradient(cross_entropy_loss(z_list[t + 1], v_pred), gradient_scale)
            r_loss += scale_gradient(cross_entropy_loss(rewards_list[t], rewards_pred), gradient_scale)

            s_state = scale_gradient(s_state, 0.5)

        v_loss /= self.config.unroll_steps + 1
        p_loss /= self.config.unroll_steps + 1
        r_loss /= self.config.unroll_steps
        return v_loss, p_loss, r_loss


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
                        "state": None,
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
                            h["state"] if i == 0 else None,
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
            n_s_state, reward = self.parameter.dynamics_network.predict(s_state, [a])
            reward = reward[0]

            if self.config.enable_rescale:
                reward = inverse_rescaling(reward)

            _, n_v = self.parameter.prediction_network.predict(n_s_state)
            n_v = n_v[0]

            s = f"{policy[a] * 100:5.1f}%"
            s += f"({int(node.visit_count):3d})(N)"
            s += f" {node.value:5.3f}(Q)"
            s += f" {node.prior:6.3f}(P)"
            s += f" {node.score:6.3f}(PUCT)"
            s += f" {reward:6.3f}(reward)"
            s += f" {n_v:6.3f}(V)"
            return s

        worker.print_discrete_action_info(worker.action, _render_sub)
