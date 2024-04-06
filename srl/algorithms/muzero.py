import logging
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import InfoType, RLBaseTypes
from srl.base.rl.config import RLConfig
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import ObservationProcessor
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker import RLWorker
from srl.rl.functions import helper
from srl.rl.functions.common import inverse_rescaling, rescaling, twohot_decode, twohot_encode
from srl.rl.memories.priority_experience_replay import (
    PriorityExperienceReplay,
    RLConfigComponentPriorityExperienceReplay,
)
from srl.rl.models.config.framework_config import RLConfigComponentFramework
from srl.rl.schedulers.scheduler import SchedulerConfig
from srl.rl.tf.blocks.alphazero_image_block import AlphaZeroImageBlock
from srl.rl.tf.blocks.input_block import create_in_block_out_image
from srl.utils.common import compare_less_version

kl = keras.layers
logger = logging.getLogger(__name__)

"""
Paper
https://arxiv.org/abs/1911.08265

Ref
https://arxiv.org/src/1911.08265v2/anc/pseudocode.py
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(
    RLConfig,
    RLConfigComponentPriorityExperienceReplay,
    RLConfigComponentFramework,
):
    """
    <:ref:`RLConfigComponentPriorityExperienceReplay`>
    <:ref:`RLConfigComponentFramework`>
    """

    #: シミュレーション回数
    num_simulations: int = 20
    #: 割引率
    discount: float = 0.99

    #: <:ref:`scheduler`> Learning rate
    lr: Union[float, SchedulerConfig] = 0.001

    #: カテゴリ化する範囲
    v_min: int = -10
    #: カテゴリ化する範囲
    v_max: int = 10

    #: policyの温度パラメータのリスト
    policy_tau: Union[float, SchedulerConfig] = 0.25

    # td_steps: int = 5  # multisteps
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
    enable_rescale: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.input_image_block.set_alphazero_block()
        self.lr = self.create_scheduler().set_linear(100_000, 0.001, 0.0001)
        self.policy_tau = self.create_scheduler()
        self.policy_tau.clear()
        self.policy_tau.add_constant(50_000, 1.0)
        self.policy_tau.add_constant(25_000, 0.5)
        self.policy_tau.add_constant(1, 0.25)

    def set_atari_config(self):
        self.num_simulations = 50
        self.batch_size = 1024
        self.memory.warmup_size = 10_000
        self.discount = 0.997
        self.lr = self.create_scheduler().set_linear(350_000, 0.05, 0.005)
        self.v_min = -300
        self.v_max = 300
        # self.td_steps = 10
        self.unroll_steps = 5
        self.policy_tau = self.create_scheduler()
        self.policy_tau.clear()
        self.policy_tau.add_constant(500_000, 1.0)
        self.policy_tau.add_constant(250_000, 0.5)
        self.policy_tau.add_constant(1, 0.25)
        self.input_image_block.set_muzero_atari_block(filters=128)
        self.dynamics_blocks = 15
        self.weight_decay = 0.0001
        self.enable_rescale = True

    def get_processors(self) -> List[Optional[ObservationProcessor]]:
        return [self.input_image_block.get_processor()]

    def get_base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.IMAGE

    def get_framework(self) -> str:
        return "tensorflow"

    def get_name(self) -> str:
        return "MuZero"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_memory()
        assert self.v_min < self.v_max
        assert self.unroll_steps > 0


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
class Memory(PriorityExperienceReplay):
    def __init__(self, *args):
        super().__init__(*args)

        self.q_min = np.inf
        self.q_max = -np.inf

    def add(self, key, batch: Any, priority: Optional[float] = None):
        if key == "memory":
            super().add(batch, priority)
        else:
            q_min, q_max = batch
            self.q_min = min(self.q_min, q_min)
            self.q_max = max(self.q_max, q_max)

    # ---------------------------------------

    def get_q(self):
        return self.q_min, self.q_max


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _RepresentationNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        # input
        self.input_block = create_in_block_out_image(
            config.input_image_block,
            config.observation_space,
        )

        # build & 出力shapeを取得
        dummy_state = np.zeros(shape=(1,) + config.observation_space.shape, dtype=np.float32)
        hidden_state = self(dummy_state)
        self.hidden_state_shape = hidden_state.shape[1:]

    def call(self, state, training=False):
        x = self.input_block(state, training=training)

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


class _DynamicsNetwork(keras.Model):
    def __init__(self, config: Config, input_shape):
        super().__init__()
        self.action_num = config.action_space.n
        v_num = config.v_max - config.v_min + 1
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
                v_num,
                activation="softmax",
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            )
        )

        # build
        self._in_shape = (h, w, ch + self.action_num)
        self.build((None,) + self._in_shape)

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

        action_image = tf.one_hot(action, self.action_num)  # (batch, action)
        action_image = tf.repeat(action_image, repeats=h * w, axis=1)  # (batch, action * h * w)
        action_image = tf.reshape(action_image, (batch_size, self.action_num, h, w))  # (batch, action, h, w)
        action_image = tf.transpose(action_image, perm=[0, 2, 3, 1])  # (batch, h, w, action)

        in_state = tf.concat([hidden_state, action_image], axis=3)
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

        return x, reward_category


class _PredictionNetwork(keras.Model):
    def __init__(self, config: Config, input_shape):
        super().__init__()
        self._in_shape = input_shape
        v_num = config.v_max - config.v_min + 1

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
                v_num,
                activation="softmax",
                kernel_regularizer=keras.regularizers.l2(config.weight_decay),
            ),
        ]

        # build
        self.build((None,) + self._in_shape)

    def call(self, state, training=False):
        policy = state
        for layer in self.policy_layers:
            policy = layer(policy, training=training)

        value = state
        for layer in self.value_layers:
            value = layer(value, training=training)

        return policy, value


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.representation_network = _RepresentationNetwork(self.config)
        # 出力shapeを取得
        hidden_state_shape = self.representation_network.hidden_state_shape

        self.prediction_network = _PredictionNetwork(self.config, hidden_state_shape)
        self.dynamics_network = _DynamicsNetwork(self.config, hidden_state_shape)

        self.q_min = np.inf
        self.q_max = -np.inf

        # cache用 (simulationで何回も使うので)
        self.P = {}
        self.V = {}

    def call_restore(self, data: Any, **kwargs) -> None:
        self.prediction_network.set_weights(data[0])
        self.dynamics_network.set_weights(data[1])
        self.representation_network.set_weights(data[2])
        self.q_min = data[3]
        self.q_max = data[4]
        self.reset_cache()

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

    # ------------------------

    def pred_PV(self, state, state_str):
        if state_str not in self.P:
            p, v_category = self.prediction_network(state)  # type:ignore , ignore check "None"
            self.P[state_str] = p[0].numpy()
            self.V[state_str] = twohot_decode(
                v_category.numpy()[0],
                abs(self.config.v_max - self.config.v_min) + 1,
                self.config.v_min,
                self.config.v_max,
            )

    def reset_cache(self):
        self.P = {}
        self.V = {}


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
def _scale_gradient(tensor, scale):
    """
    Scales the gradient for the backward pass.
    論文の疑似コードより
    """
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


class Trainer(RLTrainer[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter
        self.memory: Memory = self.memory

        self.lr_sch = SchedulerConfig.create_scheduler(self.config.lr)

        if compare_less_version(tf.__version__, "2.11.0"):
            self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch.get_rate())
        else:
            self.optimizer = keras.optimizers.legacy.Adam(learning_rate=self.lr_sch.get_rate())

    def _cross_entropy_loss(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-6, y_pred)  # log(0)回避用
        loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1)
        return loss

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        indices, batchs, weights = self.memory.sample(self.batch_size, self.train_count)

        # (batch, dict, val) -> (batch, val)
        states = []
        for b in batchs:
            states.append(b["state"])
        states = np.asarray(states)

        # (batch, dict, steps, val) -> (steps, batch, val)
        actions_list = []
        policies_list = []
        values_list = []
        rewards_list = []
        for i in range(self.config.unroll_steps + 1):
            actions = []
            policies = []
            values = []
            rewards = []
            for b in batchs:
                policies.append(b["policies"][i])
                values.append(b["values"][i])
                if i < self.config.unroll_steps:
                    actions.append(b["actions"][i])
                    rewards.append(b["rewards"][i])
            actions_list.append(actions)
            policies_list.append(np.asarray(policies).astype(np.float32))
            values_list.append(np.asarray(values).astype(np.float32))
            rewards_list.append(np.asarray(rewards).astype(np.float32))

        with tf.GradientTape() as tape:
            # --- 1st step
            hidden_states = self.parameter.representation_network(states, training=True)
            p_pred, v_pred = self.parameter.prediction_network(hidden_states, training=True)
            # loss
            policy_loss = _scale_gradient(self._cross_entropy_loss(policies_list[0], p_pred), 1.0)
            value_loss = _scale_gradient(self._cross_entropy_loss(values_list[0], v_pred), 1.0)
            reward_loss = tf.constant([0] * self.config.batch_size, dtype=tf.float32)

            # --- unroll steps
            gradient_scale = 1 / self.config.unroll_steps
            for t in range(self.config.unroll_steps):
                # pred
                hidden_states, p_rewards = self.parameter.dynamics_network.predict(
                    hidden_states, actions_list[t], training=True
                )
                p_pred, v_pred = self.parameter.prediction_network(hidden_states, training=True)

                # loss
                policy_loss += _scale_gradient(self._cross_entropy_loss(policies_list[t + 1], p_pred), gradient_scale)
                value_loss += _scale_gradient(self._cross_entropy_loss(values_list[t + 1], v_pred), gradient_scale)
                reward_loss += _scale_gradient(self._cross_entropy_loss(rewards_list[t], p_rewards), gradient_scale)

                hidden_states = _scale_gradient(hidden_states, 0.5)

            loss = tf.reduce_mean((value_loss + policy_loss + reward_loss) * weights)

            # 正則化項
            loss += tf.reduce_sum(self.parameter.representation_network.losses)
            loss += tf.reduce_sum(self.parameter.prediction_network.losses)
            loss += tf.reduce_sum(self.parameter.dynamics_network.losses)

        priorities = np.abs(value_loss.numpy())

        self.train_count += 1
        self.info = {
            "value_loss": np.mean(value_loss),
            "policy_loss": np.mean(policy_loss),
            "reward_loss": np.mean(reward_loss),
            "loss": loss.numpy(),
        }

        # lr_schedule
        if self.lr_sch.update(self.train_count):
            lr = self.lr_sch.get_rate()
            self.optimizer.learning_rate = lr
            self.info["lr"] = lr

        variables = [
            self.parameter.representation_network.trainable_variables,
            self.parameter.prediction_network.trainable_variables,
            self.parameter.dynamics_network.trainable_variables,
        ]
        grads = tape.gradient(loss, variables)
        for i in range(len(variables)):
            self.optimizer.apply_gradients(zip(grads[i], variables[i]))

        # memory update
        self.memory.update(indices, batchs, priorities)

        # 学習したらキャッシュは削除
        self.parameter.reset_cache()

        # --- 正規化用Qを保存できるように送信(remote_memory -> trainer -> parameter)
        q_min, q_max = self.memory.get_q()
        self.parameter.q_min = q_min
        self.parameter.q_max = q_max


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.policy_tau_sch = SchedulerConfig.create_scheduler(self.config.policy_tau)

        self._v_min = np.inf
        self._v_max = -np.inf

    def on_reset(self, worker) -> InfoType:
        self.history = []

        self.N = {}  # 訪問回数(s,a)
        self.W = {}  # 累計報酬(s,a)
        self.Q = {}

        return {}

    def _init_state(self, state_str):
        if state_str not in self.N:
            self.N[state_str] = [0 for _ in range(self.config.action_space.n)]
            self.W[state_str] = [0 for _ in range(self.config.action_space.n)]
            self.Q[state_str] = [0 for _ in range(self.config.action_space.n)]

    def policy(self, worker) -> Tuple[int, InfoType]:
        invalid_actions = worker.get_invalid_actions()

        # --- シミュレーションしてpolicyを作成
        self.s0 = self.parameter.representation_network(worker.state[np.newaxis, ...])
        self.s0_str = self.s0.ref()  # type:ignore , ignore check "None"
        for _ in range(self.config.num_simulations):
            self._simulation(self.s0, self.s0_str, invalid_actions)

        # 正規化用Qを保存できるように送信(remote_memory -> trainer -> parameter)
        self.memory.add("q", (self.parameter.q_min, self.parameter.q_max))

        # V
        self.state_v = self.parameter.V[self.s0_str]

        # --- 確率に比例したアクションを選択
        if not self.training:
            policy_tau = 0  # 評価時は決定的に
        else:
            policy_tau = self.policy_tau_sch.get_and_update_rate(self.total_step)

        if policy_tau == 0:
            counts = np.asarray(self.N[self.s0_str])
            action = np.random.choice(np.where(counts == counts.max())[0])
        else:
            step_policy = np.array(
                [self.N[self.s0_str][a] ** (1 / policy_tau) for a in range(self.config.action_space.n)]
            )
            step_policy /= step_policy.sum()
            action = helper.random_choice_by_probs(step_policy)

        # 学習用のpolicyはtau=1
        N = sum(self.N[self.s0_str])
        self.step_policy = [self.N[self.s0_str][a] / N for a in range(self.config.action_space.n)]

        self.action = int(action)
        return self.action, {"policy_tau": policy_tau}

    def _simulation(self, state, state_str, invalid_actions, depth: int = 0):
        if depth >= 99999:  # for safety
            return 0

        # PVを予測
        self._init_state(state_str)
        self.parameter.pred_PV(state, state_str)

        # actionを選択
        puct_list = self._calc_puct(state_str, invalid_actions, depth == 0)
        action = np.random.choice(np.where(puct_list == np.max(puct_list))[0])

        # 次の状態を取得
        n_state, reward_category = self.parameter.dynamics_network.predict(state, [action])
        n_state_str = n_state.ref()
        reward = twohot_decode(
            reward_category.numpy()[0],  # type:ignore , ignore check "None"
            abs(self.config.v_max - self.config.v_min) + 1,
            self.config.v_min,
            self.config.v_max,
        )
        enemy_turn = self.config.env_player_num > 1  # 2player以上は相手番と決め打ち

        if self.N[state_str][action] == 0:
            # leaf node ならロールアウト
            self.parameter.pred_PV(n_state, n_state_str)
            n_reward = self.parameter.V[n_state_str]
        else:
            # 子ノードに降りる(展開)
            n_reward = self._simulation(n_state, n_state_str, [], depth + 1)

        # 次が相手のターンなら、報酬は最小になってほしいので-をかける
        if enemy_turn:
            n_reward = -n_reward

        # 割引報酬
        reward = reward + self.config.discount * n_reward

        self.N[state_str][action] += 1
        self.W[state_str][action] += reward
        self.Q[state_str][action] = self.W[state_str][action] / self.N[state_str][action]

        self.parameter.q_min = min(self.parameter.q_min, self.Q[state_str][action])
        self.parameter.q_max = max(self.parameter.q_max, self.Q[state_str][action])

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
                q = self.Q[state_str][a]

                # 過去観測したQ値で正規化(MinMax)
                if self.parameter.q_min < self.parameter.q_max:
                    q = (q - self.parameter.q_min) / (self.parameter.q_max - self.parameter.q_min)

                score = q + u

                if np.isnan(score):
                    logger.warning(
                        "puct score is nan. action={}, score={}, q={}, u={}, Q={}, P={}".format(
                            a,
                            score,
                            q,
                            u,
                            self.Q[state_str],
                            self.parameter.P[state_str],
                        )
                    )
                    score = -np.inf

            scores[a] = score
        return scores

    def on_step(self, worker) -> InfoType:
        if not self.training:
            return {}

        self.history.append(
            {
                "state": worker.prev_state,
                "action": self.action,
                "policy": self.step_policy,
                "reward": worker.reward,
                "state_v": self.state_v,
            }
        )

        if worker.done:
            zero_category = twohot_encode(
                0, abs(self.config.v_max - self.config.v_min) + 1, self.config.v_min, self.config.v_max
            )

            # calc MC reward
            reward = 0
            for h in reversed(self.history):
                reward = h["reward"] + self.config.discount * reward
                h["discount_reward"] = reward

            # batch create
            for idx in range(len(self.history)):
                # --- policies
                policies = [
                    [1 / self.config.action_space.n] * self.config.action_space.n
                    for _ in range(self.config.unroll_steps + 1)
                ]
                for i in range(self.config.unroll_steps + 1):
                    if idx + i >= len(self.history):
                        break
                    policies[i] = self.history[idx + i]["policy"]

                # --- values
                values = [zero_category for _ in range(self.config.unroll_steps + 1)]
                priority = 0
                for i in range(self.config.unroll_steps + 1):
                    if idx + i >= len(self.history):
                        break
                    v = self.history[idx + i]["discount_reward"]
                    if self.config.enable_rescale:
                        v = rescaling(v)
                    priority += v - self.history[idx + i]["state_v"]
                    self._v_min = min(self._v_min, v)
                    self._v_max = max(self._v_max, v)
                    values[i] = twohot_encode(
                        v, abs(self.config.v_max - self.config.v_min) + 1, self.config.v_min, self.config.v_max
                    )
                priority /= self.config.unroll_steps + 1

                # --- actions
                actions = [random.randint(0, self.config.action_space.n - 1) for _ in range(self.config.unroll_steps)]
                for i in range(self.config.unroll_steps):
                    if idx + i >= len(self.history):
                        break
                    actions[i] = self.history[idx + i]["action"]

                # --- rewards
                rewards = [zero_category for _ in range(self.config.unroll_steps)]
                for i in range(self.config.unroll_steps):
                    if idx + i >= len(self.history):
                        break
                    r = self.history[idx + i]["reward"]
                    if self.config.enable_rescale:
                        r = rescaling(r)
                    self._v_min = min(self._v_min, r)
                    self._v_max = max(self._v_max, r)
                    rewards[i] = twohot_encode(
                        r, abs(self.config.v_max - self.config.v_min) + 1, self.config.v_min, self.config.v_max
                    )

                self.memory.add(
                    "memory",
                    {
                        "state": self.history[idx]["state"],
                        "actions": actions,
                        "policies": policies,
                        "values": values,
                        "rewards": rewards,
                    },
                    priority,
                )

        return {
            "v_min": self._v_min,
            "v_max": self._v_max,
        }

    def render_terminal(self, worker, **kwargs) -> None:
        self._init_state(self.s0_str)
        self.parameter.pred_PV(self.s0, self.s0_str)
        puct = self._calc_puct(self.s0_str, worker.prev_invalid_actions, False)
        maxa = self.action

        v = self.parameter.V[self.s0_str]
        if self.config.enable_rescale:
            v = inverse_rescaling(v)

        print(f"V_net: {v:.5f}")

        def _render_sub(a: int) -> str:
            q = self.Q[self.s0_str][a]
            c = self.N[self.s0_str][a]
            p = self.step_policy[a]

            n_s, reward_category = self.parameter.dynamics_network.predict(self.s0, [a])
            reward = twohot_decode(
                reward_category.numpy()[0],  # type:ignore , ignore check "None"
                abs(self.config.v_max - self.config.v_min) + 1,
                self.config.v_min,
                self.config.v_max,
            )
            n_s_str = n_s.ref()
            self.parameter.pred_PV(n_s, n_s_str)

            if self.config.enable_rescale:
                reward = inverse_rescaling(reward)

            s = "{:5.1f}% ({:7d})(N), {:9.5f}(Q), {:9.5f}(PUCT), {:9.5f}(P), {:9.5f}(V), {:9.5f}(reward)".format(
                p * 100,
                c,
                q,
                puct[a],
                self.parameter.P[self.s0_str][a],
                self.parameter.V[n_s_str],
                reward,
            )
            return s

        helper.render_discrete_action(int(maxa), self.config.action_space.n, worker.env, _render_sub)
