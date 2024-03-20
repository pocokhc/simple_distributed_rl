import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import InfoType, RLBaseTypes
from srl.base.env.env_run import EnvRun
from srl.base.exception import UndefinedError
from srl.base.rl.config import RLConfig
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import ObservationProcessor
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker import RLWorker
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.rl.functions import helper
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, RLConfigComponentExperienceReplayBuffer
from srl.rl.models.config.framework_config import RLConfigComponentFramework
from srl.rl.models.config.mlp_block import MLPBlockConfig
from srl.rl.models.tf import helper as helper_tf
from srl.rl.models.tf.blocks.input_block import create_in_block_out_image
from srl.rl.schedulers.scheduler import SchedulerConfig

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


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(
    RLConfig[DiscreteSpace, BoxSpace],
    RLConfigComponentExperienceReplayBuffer,
    RLConfigComponentFramework,
):
    """
    <:ref:`RLConfigComponentExperienceReplayBuffer`>
    <:ref:`RLConfigComponentFramework`>
    """

    #: シミュレーション回数
    num_simulations: int = 100
    #: 割引率
    discount: float = 1.0

    #: エピソード序盤の確率移動のステップ数
    sampling_steps: int = 1

    #: <:ref:`scheduler`> Learning rate
    lr: Union[float, SchedulerConfig] = 0.002

    #: Root prior exploration noise.
    root_dirichlet_alpha: float = 0.3
    #: Root prior exploration noise.
    root_exploration_fraction: float = 0.25

    #: PUCT
    c_base: float = 19652
    #: PUCT
    c_init: float = 1.25

    #: <:ref:`MLPBlock`> value block
    value_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())
    #: <:ref:`MLPBlock`> policy block
    policy_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())

    #: "rate" or "linear"
    value_type: str = "linear"

    def __post_init__(self):
        super().__post_init__()

        self.input_image_block.set_alphazero_block(3, 64)
        self.value_block.set((64,))
        self.policy_block.set(())

    def set_go_config(self):
        self.num_simulations = 800
        self.capacity = 500_000
        self.discount = 1.0
        self.sampling_steps = 30
        self.root_dirichlet_alpha = 0.03  # for Go, 0.3 for chess and 0.15 for shogi.
        self.root_exploration_fraction = 0.25
        self.batch_size = 4096
        self.memory.warmup_size = 10000
        self.lr = self.create_scheduler()
        self.lr.add_constant(300_000, 0.02)
        self.lr.add_constant(200_000, 0.002)
        self.lr.add_constant(1, 0.0002)
        self.input_image_block.set_alphazero_block(19, 256)
        self.value_block.set((256,))
        self.policy_block.set(())

    def get_processors(self) -> List[Optional[ObservationProcessor]]:
        return [self.input_image_block.get_processor()]

    def get_base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.IMAGE

    def get_framework(self) -> str:
        return "tensorflow"

    def get_name(self) -> str:
        return "AlphaZero"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_memory()
        self.assert_params_framework()

    def get_used_backup_restore(self) -> bool:
        return True

    def get_info_types(self) -> dict:
        return {
            "value_loss": {},
            "policy_loss": {},
            "lr": {"data": "last"},
        }


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
# network
# ------------------------------------------------------
class Network(keras.Model):
    def __init__(self, config: Config):
        super().__init__()
        self.value_type = config.value_type

        self.input_block = create_in_block_out_image(
            config.input_image_block,
            config.observation_space,
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
        self.policy_block = config.policy_block.create_block_tf()
        self.policy_out_layer = kl.Dense(
            config.action_space.n,
            activation="softmax",
            kernel_initializer="zeros",
        )

        # --- value output
        self.value_block = config.value_block.create_block_tf()
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
        self.build(helper_tf.create_batch_shape(config.observation_space.shape, (None,)))

    def call(self, x, training=False):
        x = self.input_block(x, training=training)

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


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter[Config]):
    def __init__(self, *args):
        super().__init__(*args)

        self.network = Network(self.config)

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
class Trainer(RLTrainer[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_sch = SchedulerConfig.create_scheduler(self.config.lr)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch.get_rate())

        self.value_loss = tf.zeros((1,))
        self.policy_loss = tf.zeros((1,))

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        batchs = self.memory.sample(self.batch_size)

        states = []
        policies = []
        rewards = []
        for b in batchs:
            states.append(b["state"])
            policies.append(b["policy"])
            rewards.append([b["reward"]])
        states = np.asarray(states)
        policies = np.asarray(policies, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)

        with tf.GradientTape() as tape:
            loss, self.value_loss, self.policy_loss = self.parameter.network.compute_train_loss(
                states, rewards, policies
            )
        grads = tape.gradient(loss, self.parameter.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.network.trainable_variables))

        self.train_count += 1

        # lr_schedule
        if self.lr_sch.update(self.train_count):
            lr = self.lr_sch.get_rate()
            self.optimizer.learning_rate = lr

        # 学習したらキャッシュは削除
        self.parameter.reset_cache()

    def create_info(self) -> InfoType:
        return {
            "value_loss": self.value_loss.numpy(),
            "policy_loss": self.policy_loss.numpy(),
            "lr": self.lr_sch.get_rate(),
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker[Config, Parameter, DiscreteSpace, BoxSpace]):
    def __init__(self, *args):
        super().__init__(*args)

    def on_reset(self, worker) -> dict:
        self.sampling_step = 0
        self.history = []

        self.N = {}  # 訪問回数(s,a)
        self.W = {}  # 累計報酬(s,a)

        return {}

    def _init_state(self, state_str):
        if state_str not in self.N:
            self.N[state_str] = [0 for _ in range(self.config.action_space.n)]
            self.W[state_str] = [0 for _ in range(self.config.action_space.n)]

    def policy(self, worker) -> Tuple[int, dict]:
        self.state = worker.state
        self.state_str = self.observation_space.to_str(self.state)
        self.invalid_actions = worker.get_invalid_actions()
        self._init_state(self.state_str)

        # --- シミュレーションしてpolicyを作成
        dat = worker.env.backup()
        for _ in range(self.config.num_simulations):
            self._simulation(worker.env, self.state, self.state_str)
            worker.env.restore(dat)

        # --- (教師データ) 試行回数を元に確率を計算
        N = np.sum(self.N[self.state_str])
        self.step_policy = [self.N[self.state_str][a] / N for a in range(self.config.action_space.n)]

        # --- episodeの序盤は試行回数に比例した確率でアクションを選択、それ以外は最大試行回数
        if self.sampling_step < self.config.sampling_steps:
            action = helper.random_choice_by_probs(self.N[self.state_str])
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
        n_state, rewards = self.worker.env_step(env, action)
        reward = rewards[player_index]
        n_state_str = self.observation_space.to_str(n_state)
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
            noises = np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space.n)
        else:
            noises = []

        N = np.sum(self.N[state_str])
        scores = np.zeros(self.config.action_space.n)
        for a in range(self.config.action_space.n):
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

                if np.isnan(score):
                    logger.warning(
                        "puct score is nan. action={}, score={}, q={}, u={}, P={}".format(
                            a,
                            score,
                            q,
                            u,
                            self.parameter.P[state_str],
                        )
                    )
                    score = -np.inf

            scores[a] = score
        return scores

    def on_step(self, worker) -> dict:
        self.sampling_step += 1

        if not self.training:
            return {}

        self.history.append([self.state, self.step_policy, worker.reward])

        if worker.done:
            # 報酬を逆伝搬
            reward = 0
            for state, step_policy, step_reward in reversed(self.history):
                reward = step_reward + self.config.discount * reward
                self.memory.add(
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

        helper.render_discrete_action(int(maxa), self.config.action_space.n, worker.env, _render_sub)
