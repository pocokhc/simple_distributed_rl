import logging
from dataclasses import dataclass, field
from typing import Any, List, Literal

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.env.env_run import EnvRun
from srl.base.exception import UndefinedError
from srl.base.rl.algorithms.base_dqn import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import RLProcessor
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.base.spaces.space import SpaceBase
from srl.rl import functions as funcs
from srl.rl.memories.replay_buffer import ReplayBufferConfig, RLReplayBuffer
from srl.rl.models.config.input_image_block import InputImageBlockConfig
from srl.rl.models.config.mlp_block import MLPBlockConfig
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig
from srl.rl.tf.model import KerasModelAddedSummary

kl = keras.layers
logger = logging.getLogger(__name__)

"""
Paper
AlphaGoZero: https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
AlphaZero: https://arxiv.org/abs/1712.01815
           https://www.science.org/doi/10.1126/science.aar6404

Code ref:
https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning
"""


@dataclass
class Config(RLConfig):
    #: シミュレーション回数
    num_simulations: int = 100
    #: 割引率
    discount: float = 1.0
    #: エピソード序盤の確率移動のステップ数
    sampling_steps: int = 1

    #: Batch size
    batch_size: int = 32
    #: <:ref:`ReplayBufferConfig`>
    memory: ReplayBufferConfig = field(default_factory=lambda: ReplayBufferConfig())

    #: Learning rate
    lr: float = 0.002
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())

    #: Root prior exploration noise.
    root_dirichlet_alpha: float = 0.3
    #: Root prior exploration noise.
    root_exploration_fraction: float = 0.25

    #: PUCT
    c_base: float = 19652
    #: PUCT
    c_init: float = 1.25

    #: <:ref:`InputImageBlockConfig`>
    input_image_block: InputImageBlockConfig = field(default_factory=lambda: InputImageBlockConfig().set_alphazero_block(3, 64))
    #: <:ref:`MLPBlockConfig`> value block
    value_block: MLPBlockConfig = field(default_factory=lambda: MLPBlockConfig().set((64,)))
    #: <:ref:`MLPBlockConfig`> policy block
    policy_block: MLPBlockConfig = field(default_factory=lambda: MLPBlockConfig().set(()))

    #: "rate" or "linear"
    value_type: Literal["rate", "linear"] = "linear"

    def set_go_config(self):
        self.num_simulations = 800
        self.capacity = 500_000
        self.discount = 1.0
        self.sampling_steps = 30
        self.root_dirichlet_alpha = 0.03
        self.root_exploration_fraction = 0.25
        self.batch_size = 4096
        self.memory.warmup_size = 10000
        self.lr_scheduler.set_piecewise(
            [300_000, 500_000],
            [0.02, 0.002, 0.0002],
        )
        self.input_image_block.set_alphazero_block(19, 256)
        self.value_block.set((256,))
        self.policy_block.set(())

    def set_chess_config(self):
        self.set_go_config()
        self.root_dirichlet_alpha = 0.3

    def set_shogi_config(self):
        self.set_go_config()
        self.root_dirichlet_alpha = 0.15

    def get_name(self) -> str:
        return "AlphaZero"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if not prev_observation_space.is_image():
            raise ValueError(f"The input supports only image format. {prev_observation_space}")
        return self.input_image_block.get_processors()

    def get_framework(self) -> str:
        return "tensorflow"

    def use_backup_restore(self) -> bool:
        return True


register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


class Memory(RLReplayBuffer):
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


class Network(KerasModelAddedSummary):
    def __init__(self, config: Config):
        super().__init__()
        self.value_type = config.value_type

        self.in_block = config.input_image_block.create_tf_block(
            config.observation_space,
            out_flatten=False,
        )

        # --- policy image
        self.input_image_policy_layers = [
            kl.Conv2D(
                2,
                kernel_size=(1, 1),
                strides=1,
                padding="same",
            ),
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Flatten(),
        ]

        # --- value image
        self.input_image_value_layers = [
            kl.Conv2D(
                1,
                kernel_size=(1, 1),
                strides=1,
                padding="same",
            ),
            kl.BatchNormalization(),
            kl.ReLU(),
            kl.Flatten(),
        ]

        # --- policy output
        self.policy_block = config.policy_block.create_tf_block()
        self.policy_out_layer = kl.Dense(
            config.action_space.n,
            activation="softmax",
            kernel_initializer="zeros",
        )

        # --- value output
        self.value_block = config.value_block.create_tf_block()
        if config.value_type == "rate":
            self.value_out_layer = kl.Dense(
                1,
                activation="tanh",
                kernel_initializer="truncated_normal",
            )
        elif config.value_type == "linear":
            self.value_out_layer = kl.Dense(1, kernel_initializer="truncated_normal")
        else:
            raise UndefinedError(config.value_type)

        # build
        self(np.zeros((1,) + config.observation_space.shape))

    def call(self, x, training=False):
        x = self.in_block(x, training=training)

        # --- policy image
        x1 = x
        for layer in self.input_image_policy_layers:
            x1 = layer(x1, training=training)

        # --- value image
        x2 = x
        for layer in self.input_image_value_layers:
            x2 = layer(x2, training=training)

        # --- policy output
        x1 = self.policy_block(x1, training=training)
        x1 = self.policy_out_layer(x1, training=training)

        # --- value output
        x2 = self.value_block(x2, training=training)
        x2 = self.value_out_layer(x2, training=training)

        return x1, x2

    @tf.function
    def compute_train_loss(self, state, reward, policy):
        p_pred, v_pred = self(state, training=True)

        # value: 状態に対する勝率(reward)を教師に学習(MSE)
        value_loss = tf.reduce_mean(tf.square(reward - v_pred))

        # policy: 選んだアクション(MCTSの結果)を教師に学習(categorical cross entropy)
        if self.value_type == "rate":
            p_pred = tf.clip_by_value(p_pred, 1e-10, p_pred)  # log(0)回避用
            policy_loss = -tf.reduce_mean(tf.reduce_sum(policy * tf.math.log(p_pred), axis=1))
        elif self.value_type == "linear":
            policy_loss = tf.reduce_mean(tf.square(policy - p_pred))
        else:
            raise UndefinedError(self.value_type)

        loss = value_loss + policy_loss
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss, value_loss, policy_loss


class Parameter(RLParameter[Config]):
    def setup(self):
        self.network = Network(self.config)
        self.q_min = float("inf")
        self.q_max = float("-inf")

    def call_restore(self, data: Any, **kwargs) -> None:
        self.network.set_weights(data[0])
        self.q_min = min(self.q_min, data[1])
        self.q_max = max(self.q_max, data[2])

    def call_backup(self, **kwargs):
        return [
            self.network.get_weights(),
            self.q_min,
            self.q_max,
        ]

    def summary(self, **kwargs):
        self.network.summary(**kwargs)


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        lr = self.config.lr_scheduler.apply_tf_scheduler(self.config.lr)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return

        states, policies, rewards = zip(*batches)
        states = np.asarray(states)
        policies = np.asarray(policies, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)

        with tf.GradientTape() as tape:
            loss, value_loss, policy_loss = self.parameter.network.compute_train_loss(states, rewards, policies)
        grads = tape.gradient(loss, self.parameter.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.network.trainable_variables))
        self.info["value_loss"] = value_loss.numpy()
        self.info["policy_loss"] = policy_loss.numpy()

        self.train_count += 1

        # --- 正規化用Qを保存(parameterはtrainerからしか保存されない)
        # (remote_memory -> trainer -> parameter)
        q = self.memory.get_q()
        if q is not None:
            self.parameter.q_min = min(self.parameter.q_min, q[0])
            self.parameter.q_max = max(self.parameter.q_max, q[1])


class Node:
    def __init__(self, prior: float, is_root):
        self.prior = prior
        self.is_root = is_root
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: List[Node] = []
        self.reward: float = 0.0
        self.v: float = 0.0
        self.score: float = 0.0
        self.enemy_turn: bool = False

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def expand(self, policy: List[float], v) -> None:
        self.v = v
        self.children = [Node(prior, is_root=False) for prior in policy]


class MCTS:
    def __init__(self, config: Config, parameter: Parameter) -> None:
        self.cfg = config
        self.parameter = parameter

    def simulation(self, env: EnvRun, root_state: np.ndarray, training):
        # --- root情報
        p, v = self.parameter.network(root_state[np.newaxis, ...])
        root = Node(prior=0.0, is_root=True)
        root.expand(p[0].numpy(), v[0][0].numpy())

        dat = env.backup()
        for _ in range(self.cfg.num_simulations):
            # --- 子ノードまで降りる
            node = root
            search_path = [node]
            while node.children:
                # select action
                action = self._select_action(node, env.get_invalid_actions(), training)
                node = node.children[action]
                search_path.append(node)

                # env step
                player_idx = env.next_player
                state: Any = env.step_from_rl(action, self.cfg)
                node.reward = env.rewards[player_idx]
                node.enemy_turn = player_idx != env.next_player

            if env.done:
                value = 0
            else:
                # --- expand
                p, v = self.parameter.network(state[np.newaxis, ...])
                value = v[0][0].numpy()
                node.expand(p[0].numpy(), value)

            # --- backup
            for node in reversed(search_path):
                # 相手ターンは報酬が最小になってほしいので-をかける
                if node.enemy_turn:
                    value = -value
                value = node.reward + self.cfg.discount * value
                node.value_sum += value
                node.visit_count += 1

                # 正規化用
                q = node.value
                self.parameter.q_min = min(self.parameter.q_min, q)
                self.parameter.q_max = max(self.parameter.q_max, q)

            # --- simulation last
            env.restore(dat)
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


class Worker(RLWorker[Config, Parameter, Memory]):
    def on_setup(self, worker, context) -> None:
        self.mcts = MCTS(self.config, self.parameter)

    def on_reset(self, worker):
        self.sampling_step = 0
        self.history = []
        self.root = None

    def policy(self, worker) -> int:
        # --- シミュレーションしてpolicyを作成
        root = self.mcts.simulation(worker.env, worker.state, self.training)
        if self.rendering:
            self.root = root

        # --- (教師データ) 試行回数を元に確率を計算
        action_select_count = np.array([n.visit_count for n in root.children])
        self.step_policy = action_select_count / root.visit_count

        # --- episodeの序盤は試行回数に比例した確率でアクションを選択、それ以外は最大試行回数
        if self.sampling_step < self.config.sampling_steps:
            action = funcs.random_choice_by_probs(action_select_count)
        else:
            action = np.random.choice(np.where(action_select_count == action_select_count.max())[0])

        return int(action)

    def on_step(self, worker):
        self.sampling_step += 1

        if not self.training:
            return

        # 正規化用Qを保存できるように送信(memory -> trainer -> parameter)
        self.memory.add_q(self.parameter.q_min, self.parameter.q_max)
        self.info["q_min"] = self.parameter.q_min
        self.info["q_max"] = self.parameter.q_max

        self.history.append([worker.state, self.step_policy, worker.reward])

        if worker.done:
            # calc discount reward
            reward = 0
            for state, step_policy, step_reward in reversed(self.history):
                reward = step_reward + self.config.discount * reward
                self.memory.add([state, step_policy, reward])

    def render_terminal(self, worker, **kwargs) -> None:
        if self.root is None:
            return

        print(f"V: {float(self.root.v):.5f}")

        children = self.root.children
        policy = self.step_policy

        def _render_sub(a: int) -> str:
            node = children[a]
            q = node.value

            s = f"{policy[a] * 100:5.1f}%"
            s += f"({int(node.visit_count):4d})(N)"
            s += f" {q:6.3f}(Q)"
            s += f" {node.prior:6.3f}(P)"
            s += f" {node.score:6.3f}(PUCT)"
            return s

        worker.print_discrete_action_info(worker.action, _render_sub)
