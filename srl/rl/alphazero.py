import pickle
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, List, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from srl.base.define import RLObservationType
from srl.base.env.base import EnvRun
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig
from srl.base.rl.algorithms.modelbase import ModelBaseWorker
from srl.base.rl.base import RLParameter, RLRemoteMemory, RLTrainer, WorkerRun
from srl.base.rl.registration import register
from srl.rl.functions.common import random_choice_by_probs, render_discrete_action, to_str_observation
from srl.rl.models.alphazero_image_block import AlphaZeroImageBlock
from srl.rl.models.input_layer import create_input_layer
from srl.rl.models.mlp_block import MLPBlock
from tensorflow.keras import layers as kl

"""
Paper
AlphaGoZero: https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
AlphaZero: https://arxiv.org/abs/1712.01815
           https://www.science.org/doi/10.1126/science.aar6404
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    simulation_times: int = 100
    capacity: int = 10_000
    gamma: float = 1.0  # 割引率

    sampling_steps: int = 10
    batch_size: int = 64
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
    cnn_block: kl.Layer = AlphaZeroImageBlock
    cnn_block_kwargs: dict = None
    value_block: kl.Layer = MLPBlock
    value_block_kwargs: dict = None
    policy_block: kl.Layer = MLPBlock
    policy_block_kwargs: dict = None

    def set_go_config(self):
        self.simulation_times = 800
        self.capacity = 500_000
        self.gamma = 1.0
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

        self.cnn_block = AlphaZeroImageBlock
        self.cnn_block_kwargs = dict(n_blocks=19, filters=256)
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
        if self.cnn_block_kwargs is None:
            self.cnn_block_kwargs = dict(n_blocks=3, filters=64)
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
class RemoteMemory(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.buffer_nw = deque(maxlen=self.config.capacity)
        self.buffer_count = []

    def length(self) -> int:
        return len(self.buffer_nw) + len(self.buffer_count)

    def restore(self, data: Any) -> None:
        self.buffer_nw = data[0]
        self.buffer_count = data[1]

    def backup(self):
        return [
            self.buffer_nw,
            self.buffer_count,
        ]

    # ---------------------------
    def add_nw(self, batch: Any):
        self.buffer_nw.append(batch)

    def sample_nw(self, batch_size: int):
        return random.sample(self.buffer_nw, batch_size)

    def length_nw(self):
        return len(self.buffer_nw)

    def add_count(self, batch: Any):
        self.buffer_count.append(batch)

    def sample_count(self):
        batchs = self.buffer_count
        self.buffer_count = []
        return batchs

    def length_count(self):
        return len(self.buffer_count)


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _Network(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        in_state, c, use_image_head = create_input_layer(
            config.observation_shape,
            config.env_observation_type,
        )
        if use_image_head:
            c = config.cnn_block(**config.cnn_block_kwargs)(c)

            # --- policy image
            c1 = kl.Conv2D(2, kernel_size=1, padding="same")(c)
            c1 = kl.BatchNormalization()(c1)
            c1 = kl.Activation("relu")(c1)
            c1 = kl.Flatten()(c1)

            # --- value image
            c2 = kl.Conv2D(1, kernel_size=1, padding="same")(c)
            c2 = kl.BatchNormalization()(c2)
            c2 = kl.Activation("relu")(c2)
            c2 = kl.Flatten()(c2)
        else:
            c1 = c
            c2 = c

        # --- policy output
        c1 = config.policy_block(**config.policy_block_kwargs)(c1)
        policy = kl.Dense(config.action_num, activation="softmax", bias_initializer="he_normal")(c1)

        # --- value output
        c2 = config.value_block(**config.value_block_kwargs)(c2)
        value = kl.Dense(1, activation="tanh")(c2)

        self.model = keras.Model(in_state, [policy, value], name="PVNetwork")

        # 重みを初期化
        dummy_state = np.zeros(shape=(1,) + config.observation_shape, dtype=float)
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

        self.N = {}  # 訪問回数
        self.W = {}  # 累計報酬

        # cache用 (simulationで何回も使うので)
        self.P = {}
        self.V = {}

    def restore(self, data: Any) -> None:
        self.network.set_weights(pickle.loads(data))
        self.reset_cache()

    def backup(self):
        return pickle.dumps(self.network.get_weights())

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

    def init_state(self, state_str):
        if state_str not in self.N:
            self.N[state_str] = [0 for _ in range(self.config.action_num)]
            self.W[state_str] = [0 for _ in range(self.config.action_num)]


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
        self.value_loss = keras.losses.MeanSquaredError()
        self.policy_loss = keras.losses.CategoricalCrossentropy()

        self.train_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self) -> dict:
        info_nw = self._train_nw()
        info_count = self._train_count()

        info_nw.update(info_count)
        return info_nw

    def _train_nw(self) -> dict:
        if self.remote_memory.length_nw() < self.config.warmup_size:
            return {}
        batchs = self.remote_memory.sample_nw(self.config.batch_size)

        states = []
        policies = []
        rewards = []
        for b in batchs:
            states.append(b["state"])
            policies.append(b["policy"])
            rewards.append(b["reward"])
        states = np.asarray(states)
        policies = np.asarray(policies)
        rewards = np.asarray(rewards).reshape((-1, 1))

        with tf.GradientTape() as tape:
            p_pred, v_pred = self.parameter.network(states)

            # value: 状態に対する勝率(reward)を教師に学習
            value_loss = self.value_loss(rewards, v_pred)

            # policy: 選んだアクション(MCTSの結果)を教師に学習
            policy_loss = self.policy_loss(policies, p_pred)

            loss = tf.reduce_mean(value_loss + policy_loss)

        grads = tape.gradient(loss, self.parameter.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.network.trainable_variables))

        self.train_count += 1

        # lr_schedule
        if self.train_count in self.lr_schedule:
            self.optimizer.learning_rate = self.lr_schedule[self.train_count]

        # 学習したらキャッシュは削除
        self.parameter.reset_cache()

        return {
            "value_loss": value_loss.numpy(),
            "policy_loss": policy_loss.numpy(),
            "rl": self.optimizer.learning_rate.numpy(),
        }

    def _train_count(self) -> dict:
        if self.remote_memory.length_count() < 1:
            return {}
        for b in self.remote_memory.sample_count():
            state_str = b["state_str"]
            action = b["action"]
            reward = b["reward"]
            self.parameter.init_state(state_str)
            self.parameter.N[state_str][action] += 1
            self.parameter.W[state_str][action] += reward
        return {"N_size": len(self.parameter.N)}


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

    def call_policy(self, state: np.ndarray, env: EnvRun, worker: WorkerRun) -> int:
        self.state = state
        self.state_str = to_str_observation(state)
        self.invalid_actions = env.get_invalid_actions(self.player_index)
        self.parameter.init_state(self.state_str)

        # シミュレーションしてpolicyを作成
        dat = env.backup()
        for _ in range(self.config.simulation_times):
            self._simulation(env, state, self.state_str, self.invalid_actions)
            env.restore(dat)
        N = sum(self.parameter.N[self.state_str])
        n = self.parameter.N[self.state_str]
        policy = [n[a] / N for a in range(self.config.action_num)]

        if self.step < self.config.sampling_steps:
            # episodeの序盤は試行回数に比例した確率でアクションを選択
            probs = np.array([0 if a in self.invalid_actions else v for a, v in enumerate(policy)])  # mask
            action = random_choice_by_probs(probs)
        else:
            action = random.choice(np.where(policy == np.max(policy))[0])

        self.step_policy = policy
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
        self.parameter.init_state(state_str)
        self.parameter.pred_PV(state, state_str)

        # actionを選択
        puct_list = self._calc_puct(state_str, invalid_actions, depth == 0)
        action = random.choice(np.where(puct_list == np.max(puct_list))[0])

        # 1step
        n_state, rewards, done = self.env_step(env, action)
        reward = rewards[player_index]
        n_state_str = to_str_observation(n_state)

        if done:
            pass  # 終了
        elif self.parameter.N[state_str][action] == 0:
            # leaf node ならロールアウト
            self.parameter.pred_PV(n_state, n_state_str)
            # 一応割り引く
            reward = reward + self.config.gamma * self.parameter.V[n_state_str]
        else:
            n_invalid_actions = self.get_invalid_actions(env)

            enemy_turn = player_index != env.next_player_index

            # 子ノードに降りる(展開)
            n_reward = self._simulation(env, n_state, n_state_str, n_invalid_actions, depth + 1)

            # 次が相手のターンの報酬は最小になってほしいので-をかける
            if enemy_turn:
                n_reward = -n_reward

            # 割引報酬
            reward = reward + self.config.gamma * n_reward

        # 結果を記録
        self.parameter.N[state_str][action] += 1
        self.parameter.W[state_str][action] += reward
        if self.distributed:
            self.remote_memory.add_count(
                {
                    "state_str": state_str,
                    "action": action,
                    "reward": reward,
                }
            )

        return reward

    def _calc_puct(self, state_str, invalid_actions, is_root):

        # ディリクレノイズ
        if is_root:
            noises = np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_num)

        N = np.sum(self.parameter.N[state_str])
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

                n = self.parameter.N[state_str][a]
                c = np.log((1 + N + self.config.c_base) / self.config.c_base) + self.config.c_init
                u = c * P * (np.sqrt(N) / (1 + n))
                q = 0 if self.parameter.N[state_str][a] == 0 else self.parameter.W[state_str][a] / n
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

        # value networkの想定が-1～1(tanh)なのでclip
        if reward < -1:
            reward = -1
        elif reward > 1:
            reward = 1

        self.history.append([self.state, self.step_policy, reward])

        if done:
            # 報酬を逆伝搬
            reward = 0
            for h in reversed(self.history):
                reward = h[2] + self.config.gamma * reward
                self.remote_memory.add_nw(
                    {
                        "state": h[0],
                        "policy": h[1],
                        "reward": reward,
                    }
                )

        return {}

    def call_render(self, env: EnvRun, worker: WorkerRun) -> None:
        self.parameter.init_state(self.state_str)
        self.parameter.pred_PV(self.state, self.state_str)
        print(f"value: {self.parameter.V[self.state_str]:7.3f}")
        puct = self._calc_puct(self.state_str, self.invalid_actions, False)

        def _render_sub(a: int) -> str:
            if self.state_str in self.parameter.W:
                q = self.parameter.W[self.state_str][a]
                c = self.parameter.N[self.state_str][a]
                if c != 0:
                    q = q / c
            else:
                q = 0
                c = 0
            s = "{}: Q {:9.5f}({:7d}), policy {:9.5f}, puct {:.5f}".format(
                env.action_to_str(a),
                q,
                c,
                self.parameter.P[self.state_str][a],
                puct[a],
            )
            return s

        render_discrete_action(self.invalid_actions, None, env, _render_sub)
