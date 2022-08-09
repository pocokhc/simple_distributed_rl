import random
from dataclasses import dataclass
from typing import Any, List, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from srl.base.define import RLObservationType
from srl.base.env.base import EnvRun
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig
from srl.base.rl.algorithms.modelbase import ModelBaseWorker
from srl.base.rl.base import RLParameter, RLTrainer, WorkerRun
from srl.base.rl.registration import register
from srl.base.rl.remote_memory.experience_replay_buffer import ExperienceReplayBuffer
from srl.rl.functions.common import random_choice_by_probs, render_discrete_action, to_str_observation
from srl.rl.models.alphazero_image_block import AlphaZeroImageBlock
from srl.rl.models.input_layer import create_input_layer
from srl.rl.models.mlp_block import MLPBlock
from tensorflow.keras import layers as kl
from tensorflow.keras import regularizers

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
class Config(DiscreteActionConfig):

    simulation_times: int = 100
    capacity: int = 10_000
    discount: float = 1.0  # 割引率

    sampling_steps: int = 1
    batch_size: int = 128
    warmup_size: int = 1000

    # 学習率
    lr_schedule: List[dict] = None

    # Root prior exploration noise.
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25

    # PUCT
    c_base: float = 19652
    c_init: float = 1.25

    # model
    input_image_block: kl.Layer = AlphaZeroImageBlock
    input_image_block_kwargs: dict = None
    value_block: kl.Layer = MLPBlock
    value_block_kwargs: dict = None
    policy_block: kl.Layer = MLPBlock
    policy_block_kwargs: dict = None

    def set_go_config(self):
        self.simulation_times = 800
        self.capacity = 500_000
        self.discount = 1.0
        self.sampling_steps = 30
        self.root_dirichlet_alpha = 0.03  # for Go, 0.3 for chess and 0.15 for shogi.
        self.root_exploration_fraction = 0.25
        self.batch_size = 4096
        self.warmup_size = 10000
        self.lr_schedule = [
            {"train": 0, "lr": 0.02},
            {"train": 300_000, "lr": 0.002},
            {"train": 500_000, "lr": 0.0002},
        ]

        self.input_image_block = AlphaZeroImageBlock
        self.input_image_block_kwargs = dict(n_blocks=19, filters=256)
        self.value_block = MLPBlock
        self.value_block_kwargs = dict(hidden_layer_sizes=(256,))
        self.policy_block = MLPBlock
        self.policy_block_kwargs = dict(hidden_layer_sizes=())

    def __post_init__(self):
        super().__init__()
        if self.lr_schedule is None:
            self.lr_schedule = [
                {"train": 0, "lr": 0.2},
                {"train": 100, "lr": 0.02},
                {"train": 500, "lr": 0.002},
                {"train": 1000, "lr": 0.0002},
            ]
        if self.input_image_block_kwargs is None:
            self.input_image_block_kwargs = dict(n_blocks=3, filters=64)
        if self.value_block_kwargs is None:
            self.value_block_kwargs = dict(hidden_layer_sizes=(64,))
        if self.policy_block_kwargs is None:
            self.policy_block_kwargs = dict(hidden_layer_sizes=())

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.CONTINUOUS

    @staticmethod
    def getName() -> str:
        return "AlphaZero"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.batch_size < self.warmup_size
        assert self.lr_schedule[0]["train"] == 0


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
# network
# ------------------------------------------------------
class _Network(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        if "l2" in config.input_image_block_kwargs:
            _l2 = config.input_image_block_kwargs["l2"]
        else:
            _l2 = 0.0001

        in_state, c, use_image_head = create_input_layer(
            config.observation_shape,
            config.env_observation_type,
        )
        if use_image_head:
            c = config.input_image_block(**config.input_image_block_kwargs)(c)

            # --- policy image
            c1 = kl.Conv2D(
                2,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                kernel_regularizer=regularizers.l2(_l2),
            )(c)
            c1 = kl.BatchNormalization()(c1)
            c1 = kl.LeakyReLU()(c1)
            c1 = kl.Flatten()(c1)

            # --- value image
            c2 = kl.Conv2D(
                1,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                kernel_regularizer=regularizers.l2(_l2),
            )(c)
            c2 = kl.BatchNormalization()(c2)
            c2 = kl.LeakyReLU()(c2)
            c2 = kl.Flatten()(c2)
        else:
            c1 = c
            c2 = c

        # --- policy output
        c1 = config.policy_block(**config.policy_block_kwargs)(c1)
        policy = kl.Dense(
            config.action_num,
            activation="softmax",
            kernel_initializer="truncated_normal",
        )(c1)

        # --- value output
        c2 = config.value_block(**config.value_block_kwargs)(c2)
        value = kl.Dense(
            1,
            # activation="tanh",  # 論文はtanh(-1～1)
            kernel_initializer="truncated_normal",
        )(c2)

        self.model = keras.Model(in_state, [policy, value], name="PVNetwork")

        # 重みを初期化
        dummy_state = np.zeros(shape=(1,) + config.observation_shape, dtype=np.float32)
        policy, value = self(dummy_state)
        assert policy.shape == (1, config.action_num)
        assert value.shape == (1, 1)

    def call(self, state):
        return self.model(state)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.network = _Network(self.config)

        # cache用 (simulationで何回も使うので)
        self.P = {}
        self.V = {}

    def restore(self, data: Any) -> None:
        self.network.set_weights(data)
        self.reset_cache()

    def backup(self):
        return self.network.get_weights()

    def summary(self, **kwargs):
        self.network.model.summary(**kwargs)

    # ------------------------

    def pred_PV(self, state, state_str):
        if state_str not in self.P:
            p, v = self.network(np.asarray([state]))
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
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        # lr_schedule
        self.lr_schedule = {}
        for lr_list in self.config.lr_schedule:
            self.lr_schedule[lr_list["train"]] = lr_list["lr"]

        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_schedule[0])

        self.train_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self) -> dict:
        if self.remote_memory.length() < self.config.warmup_size:
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
            p_pred, v_pred = self.parameter.network(states)

            # value: 状態に対する勝率(reward)を教師に学習(MSE)
            value_loss = tf.square(rewards - v_pred)

            # policy: 選んだアクション(MCTSの結果)を教師に学習(categorical cross entropy)
            p_pred = tf.clip_by_value(p_pred, 1e-6, p_pred)  # log(0)回避用
            policy_loss = -tf.reduce_sum(policies * tf.math.log(p_pred), axis=1, keepdims=True)

            loss = tf.reduce_mean(value_loss + policy_loss)
            loss += tf.reduce_sum(self.parameter.network.losses)  # 正則化のLoss

        grads = tape.gradient(loss, self.parameter.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.network.trainable_variables))

        self.train_count += 1

        # lr_schedule
        if self.train_count in self.lr_schedule:
            self.optimizer.learning_rate = self.lr_schedule[self.train_count]

        # 学習したらキャッシュは削除
        self.parameter.reset_cache()

        return {
            "value_loss": np.mean(value_loss.numpy()),
            "policy_loss": np.mean(policy_loss.numpy()),
            "loss": loss.numpy(),
            "lr": self.optimizer.learning_rate.numpy(),
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(ModelBaseWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

    def call_on_reset(self, state: np.ndarray, env: EnvRun, worker: WorkerRun) -> None:
        self.step = 0
        self.history = []

        self.N = {}  # 訪問回数(s,a)
        self.W = {}  # 累計報酬(s,a)

    def _init_state(self, state_str):
        if state_str not in self.N:
            self.N[state_str] = [0 for _ in range(self.config.action_num)]
            self.W[state_str] = [0 for _ in range(self.config.action_num)]

    def call_policy(self, state: np.ndarray, env: EnvRun, worker: WorkerRun) -> int:
        self.state = state
        self.state_str = to_str_observation(state)
        self.invalid_actions = env.get_invalid_actions()
        self._init_state(self.state_str)

        # --- シミュレーションしてpolicyを作成
        dat = env.backup()
        for _ in range(self.config.simulation_times):
            self._simulation(env, state, self.state_str, self.invalid_actions)
            env.restore(dat)

        # --- (教師データ) 試行回数を元に確率を計算
        N = np.sum(self.N[self.state_str])
        self.step_policy = [self.N[self.state_str][a] / N for a in range(self.config.action_num)]

        # --- episodeの序盤は試行回数に比例した確率でアクションを選択、それ以外は最大試行回数
        if self.step < self.config.sampling_steps:
            action = random_choice_by_probs(self.N[self.state_str])
        else:
            counts = np.asarray(self.N[self.state_str])
            action = random.choice(np.where(counts == counts.max())[0])

        return int(action)

    def _simulation(
        self,
        env: EnvRun,
        state: np.ndarray,
        state_str: str,
        invalid_actions,
        depth: int = 0,
    ):
        if depth >= env.max_episode_steps:  # for safety
            return 0

        player_index = env.next_player_index

        # PVを予測
        self._init_state(state_str)
        self.parameter.pred_PV(state, state_str)

        # actionを選択
        puct_list = self._calc_puct(state_str, invalid_actions, depth == 0)
        action = int(random.choice(np.where(puct_list == np.max(puct_list))[0]))

        # 1step
        n_state, rewards, done = self.env_step(env, action)
        reward = rewards[player_index]
        n_state_str = to_str_observation(n_state)
        enemy_turn = player_index != env.next_player_index

        if done:
            n_reward = 0
        elif self.N[state_str][action] == 0:
            # leaf node ならロールアウト
            self.parameter.pred_PV(n_state, n_state_str)
            n_reward = self.parameter.V[n_state_str]
        else:
            n_invalid_actions = self.get_invalid_actions(env)

            # 子ノードに降りる(展開)
            n_reward = self._simulation(env, n_state, n_state_str, n_invalid_actions, depth + 1)

        # 次が相手のターンなら、報酬は最小になってほしいので-をかける
        if enemy_turn:
            n_reward = -n_reward

        # 割引報酬
        reward = reward + self.config.discount * n_reward

        # 結果を記録
        self.N[state_str][action] += 1
        self.W[state_str][action] += reward

        return reward

    def _calc_puct(self, state_str, invalid_actions, is_root):

        # ディリクレノイズ
        if is_root:
            noises = np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_num)

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

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        env: EnvRun,
        worker: WorkerRun,
    ):
        self.step += 1

        if not self.training:
            return {}

        self.history.append([self.state, self.step_policy, reward])

        if done:
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

    def render_terminal(self, env, worker, **kwargs) -> None:
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

        render_discrete_action(self.invalid_actions, maxa, env, _render_sub)
