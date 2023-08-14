from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import RLTypes
from srl.base.env.env_run import EnvRun
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.config import RLConfig
from srl.base.rl.registration import register
from srl.base.rl.worker_rl import RLWorker
from srl.base.rl.worker_run import WorkerRun
from srl.rl.functions.common import random_choice_by_probs, render_discrete_action, to_str_observation
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, ExperienceReplayBufferConfig
from srl.rl.models.alphazero_block import AlphaZeroBlockConfig
from srl.rl.models.mlp_block import MLPBlockConfig
from srl.rl.models.tf.input_block import InputBlock
from srl.rl.schedulers.scheduler import SchedulerConfig

kl = keras.layers

"""
Paper
AlphaGoZero: https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
AlphaZero: https://arxiv.org/abs/1712.01815
           https://www.science.org/doi/10.1126/science.aar6404

Code ref:
https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig, ExperienceReplayBufferConfig):
    num_simulations: int = 100
    discount: float = 1.0

    sampling_steps: int = 1
    batch_size: int = 128
    memory_warmup_size: int = 1000

    # 学習率
    lr: SchedulerConfig = field(init=False, default_factory=lambda: SchedulerConfig())

    # Root prior exploration noise.
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25

    # PUCT
    c_base: float = 19652
    c_init: float = 1.25

    # model
    input_image_block: AlphaZeroBlockConfig = field(init=False, default_factory=lambda: AlphaZeroBlockConfig())
    value_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())
    policy_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())

    def __post_init__(self):
        super().__post_init__()

        self.lr.clear()
        self.lr.add_constant(100, 0.02)
        self.lr.add_constant(1000, 0.002)
        self.lr.add_constant(1, 0.0002)

        self.input_image_block.set_alphazero_block(3, 64)
        self.value_block.set_mlp((64,))
        self.policy_block.set_mlp(())

    def set_go_config(self):
        self.num_simulations = 800
        self.capacity = 500_000
        self.discount = 1.0
        self.sampling_steps = 30
        self.root_dirichlet_alpha = 0.03  # for Go, 0.3 for chess and 0.15 for shogi.
        self.root_exploration_fraction = 0.25
        self.batch_size = 4096
        self.memory_warmup_size = 10000
        self.lr.clear()
        self.lr.add_constant(300_000, 0.02)
        self.lr.add_constant(200_000, 0.002)
        self.lr.add_constant(1, 0.0002)
        self.input_image_block.set_alphazero_block(19, 256)
        self.value_block.set_mlp((256,))
        self.policy_block.set_mlp(())

    @property
    def base_action_type(self) -> RLTypes:
        return RLTypes.DISCRETE

    @property
    def base_observation_type(self) -> RLTypes:
        return RLTypes.CONTINUOUS

    def get_use_framework(self) -> str:
        return "tensorflow"

    def getName(self) -> str:
        return "AlphaZero"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.batch_size <= self.memory_warmup_size

    @property
    def info_types(self) -> dict:
        return {
            "value_loss": {},
            "policy_loss": {},
            "lr": {"data": "last"},
        }


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
    pass


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _Network(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        # --- in block
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)
        self.use_image_layer = self.in_block.use_image_layer

        # --- image
        if self.use_image_layer:
            self.input_image_block = config.input_image_block.create_block_tf()

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
        self.policy_block = config.policy_block.create_block_tf()
        self.policy_out_layer = kl.Dense(
            config.action_num,
            activation="softmax",
            kernel_initializer="truncated_normal",
        )

        # --- value output
        self.value_block = config.value_block.create_block_tf()
        self.value_out_layer = kl.Dense(
            1,
            activation="tanh",  # 論文はtanh(-1～1)
            kernel_initializer="truncated_normal",
        )

        # build
        self.build((None,) + config.observation_shape)

    def call(self, state, training=False):
        # input
        x = self.in_block(state, training=training)
        if self.use_image_layer:
            x = self.input_image_block(x, training=training)

            # --- policy image
            x1 = x
            for layer in self.input_image_policy_layers:
                x1 = layer(x1, training=training)

            # --- value image
            x2 = x
            for layer in self.input_image_value_layers:
                x2 = layer(x2, training=training)
        else:
            x1 = x
            x2 = x

        # --- policy output
        x1 = self.policy_block(x1, training=training)
        x1 = self.policy_out_layer(x1, training=training)

        # --- value output
        x2 = self.value_block(x2, training=training)
        x2 = self.value_out_layer(x2, training=training)

        return x1, x2

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def summary(self, name="", **kwargs):
        if hasattr(self.in_block, "init_model_graph"):
            self.in_block.init_model_graph()
        if self.use_image_layer and hasattr(self.input_image_block, "init_model_graph"):
            self.input_image_block.init_model_graph()
        if hasattr(self.policy_block, "init_model_graph"):
            self.policy_block.init_model_graph()
        if hasattr(self.value_block, "init_model_graph"):
            self.value_block.init_model_graph()

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
        self.config: Config = self.config

        self.network = _Network(self.config)

        # cache用 (simulationで何回も使うので)
        self.P = {}
        self.V = {}

    def call_restore(self, data: Any, **kwargs) -> None:
        self.network.set_weights(data)
        self.reset_cache()

    def call_backup(self, **kwargs):
        return self.network.get_weights()

    def summary(self, **kwargs):
        self.network.summary(**kwargs)

    # ------------------------

    def pred_PV(self, state, state_str):
        if state_str not in self.P:
            p, v = self.network(np.asarray([state]))  # type:ignore , ignore check "None"
            self.P[state_str] = p[0].numpy()
            self.V[state_str] = v[0][0].numpy()

    def reset_cache(self):
        self.P = {}
        self.V = {}


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter
        self.remote_memory: RemoteMemory = self.remote_memory

        self.lr_sch = self.config.lr.create_schedulers()

        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch.get_rate(0))

        self.train_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self) -> dict:
        if self.remote_memory.length() < self.config.memory_warmup_size:
            return {}
        batchs = self.remote_memory.sample(self.config.batch_size)

        states = []
        policies = []
        rewards = []
        for b in batchs:
            states.append(b["state"])
            policies.append(b["policy"])
            rewards.append([b["reward"]])
        states = np.asarray(states)
        policies = np.asarray(policies)
        rewards = np.asarray(rewards)

        with tf.GradientTape() as tape:
            p_pred, v_pred = self.parameter.network(states, training=True)  # type:ignore , ignore check "None"

            # value: 状態に対する勝率(reward)を教師に学習(MSE)
            value_loss: Any = tf.square(rewards - v_pred)

            # policy: 選んだアクション(MCTSの結果)を教師に学習(categorical cross entropy)
            p_pred = tf.clip_by_value(p_pred, 1e-6, p_pred)  # log(0)回避用
            policy_loss = -tf.reduce_sum(policies * tf.math.log(p_pred), axis=1, keepdims=True)

            loss = tf.reduce_mean(value_loss + policy_loss)
            loss += tf.reduce_sum(self.parameter.network.losses)  # 正則化のLoss

        grads = tape.gradient(loss, self.parameter.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.network.trainable_variables))

        # lr_schedule
        lr = self.lr_sch.get_rate(self.train_count)
        self.optimizer.learning_rate = lr

        # 学習したらキャッシュは削除
        self.parameter.reset_cache()

        self.train_count += 1
        return {
            "value_loss": np.mean(value_loss.numpy()),
            "policy_loss": np.mean(policy_loss.numpy()),
            "lr": lr,
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter
        self.remote_memory: RemoteMemory = self.remote_memory

    def call_on_reset(self, worker: WorkerRun) -> dict:
        self.sampling_step = 0
        self.history = []

        self.N = {}  # 訪問回数(s,a)
        self.W = {}  # 累計報酬(s,a)

        return {}

    def _init_state(self, state_str):
        if state_str not in self.N:
            self.N[state_str] = [0 for _ in range(self.config.action_num)]
            self.W[state_str] = [0 for _ in range(self.config.action_num)]

    def call_policy(self, worker: WorkerRun) -> Tuple[int, dict]:
        self.state = worker.state
        self.state_str = to_str_observation(self.state, self.config.env_observation_type)
        self.invalid_actions = worker.get_invalid_actions()
        self._init_state(self.state_str)

        # --- シミュレーションしてpolicyを作成
        dat = worker.env.backup()
        for _ in range(self.config.num_simulations):
            self._simulation(worker.env, self.state, self.state_str)
            worker.env.restore(dat)

        # --- (教師データ) 試行回数を元に確率を計算
        N = np.sum(self.N[self.state_str])
        self.step_policy = [self.N[self.state_str][a] / N for a in range(self.config.action_num)]

        # --- episodeの序盤は試行回数に比例した確率でアクションを選択、それ以外は最大試行回数
        if self.sampling_step < self.config.sampling_steps:
            action = random_choice_by_probs(self.N[self.state_str])
        else:
            counts = np.asarray(self.N[self.state_str])
            action = np.random.choice(np.where(counts == counts.max())[0])

        return int(action), {}

    def _simulation(self, env: EnvRun, state: np.ndarray, state_str: str, depth: int = 0):
        if depth >= env.max_episode_steps:  # for safety
            return 0

        # PVを予測
        self._init_state(state_str)
        self.parameter.pred_PV(state, state_str)

        # actionを選択
        puct_list = self._calc_puct(state_str, env.get_invalid_actions(), depth == 0)
        action = int(np.random.choice(np.where(puct_list == np.max(puct_list))[0]))

        # 1step
        player_index = env.next_player_index
        n_state, rewards = self.worker_run.env_step(env, action)
        reward = rewards[player_index]
        n_state_str = to_str_observation(n_state)
        enemy_turn = player_index != env.next_player_index

        if env.done:
            n_value = 0
        elif self.N[state_str][action] == 0:
            # leaf node ならロールアウト
            self.parameter.pred_PV(n_state, n_state_str)
            n_value = self.parameter.V[n_state_str]
        else:
            # 子ノードに降りる(展開)
            n_value = self._simulation(env, n_state, n_state_str, depth + 1)

        # 次が相手のターンなら、報酬は最小になってほしいので-をかける
        if enemy_turn:
            n_value = -n_value

        # 割引報酬
        reward = reward + self.config.discount * n_value

        # 結果を記録
        self.N[state_str][action] += 1
        self.W[state_str][action] += reward

        return reward

    def _calc_puct(self, state_str, invalid_actions, is_root):
        # ディリクレノイズ
        if is_root:
            noises = np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_num)
        else:
            noises = []

        N = np.sum(self.N[state_str])
        scores = np.zeros(self.config.action_num)
        for a in range(self.config.action_num):
            if a in invalid_actions:
                score = -np.inf
            else:
                # P(s,a): 過去のMCTSの結果を教師あり学習した結果
                # U(s,a) = C(s) * P(s,a) * sqrt(N(s)) / (1+N(s,a))
                # C(s) = log((1+N(s)+base)/base) + c_init
                # score = Q(s,a) + U(s,a)
                P = self.parameter.P[state_str][a]

                # rootはディリクレノイズを追加
                if is_root:
                    e = self.config.root_exploration_fraction
                    P = (1 - e) * P + e * noises[a]

                n = self.N[state_str][a]
                c = np.log((1 + N + self.config.c_base) / self.config.c_base) + self.config.c_init
                u = c * P * (np.sqrt(N) / (1 + n))
                q = 0 if n == 0 else self.W[state_str][a] / n
                score = q + u
            scores[a] = score
        return scores

    def call_on_step(self, worker: WorkerRun) -> dict:
        self.sampling_step += 1

        if not self.training:
            return {}

        self.history.append([self.state, self.step_policy, worker.reward])

        if worker.done:
            # 報酬を逆伝搬
            reward = 0
            for state, step_policy, step_reward in reversed(self.history):
                reward = step_reward + self.config.discount * reward
                self.remote_memory.add(
                    {
                        "state": state,
                        "policy": step_policy,
                        "reward": reward,
                    }
                )

        return {}

    def render_terminal(self, worker, **kwargs) -> None:
        self._init_state(self.state_str)
        self.parameter.pred_PV(self.state, self.state_str)
        puct = self._calc_puct(self.state_str, self.invalid_actions, False)
        maxa = np.argmax(self.N[self.state_str])
        N = np.sum(self.N[self.state_str])

        print(f"V_net: {self.parameter.V[self.state_str]:.5f}")

        def _render_sub(a: int) -> str:
            if self.state_str in self.W:
                q = self.W[self.state_str][a]
                c = self.N[self.state_str][a]
                if c != 0:
                    q = q / c
                if N == 0:
                    p = 0
                else:
                    p = (self.N[self.state_str][a] / N) * 100
            else:
                p = 0
                q = 0
                c = 0

            s = "{:5.1f}% ({:7d})(N), {:9.5f}(Q), {:9.5f}(P_net), {:.5f}(PUCT)".format(
                p,
                c,
                q,
                self.parameter.P[self.state_str][a],
                puct[a],
            )
            return s

        render_discrete_action(maxa, worker.env, self.config, _render_sub)
