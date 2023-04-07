import logging
import random
from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as kl
from tensorflow.keras import regularizers

from srl.base.define import EnvObservationType, RLObservationType
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.processor import Processor
from srl.base.rl.processors.image_processor import ImageProcessor
from srl.base.rl.registration import register
from srl.base.rl.remote_memory.priority_experience_replay import PriorityExperienceReplay
from srl.rl.functions.common import (
    float_category_decode,
    float_category_encode,
    inverse_rescaling,
    random_choice_by_probs,
    render_discrete_action,
    rescaling,
)
from srl.rl.models.tf.alphazero_image_block import AlphaZeroImageBlock
from srl.rl.models.tf.input_layer import create_input_layer
from srl.rl.models.tf.muzero_atari_block import MuZeroAtariBlock
from srl.utils.common import compare_less_version

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
class Config(DiscreteActionConfig):

    num_simulations: int = 20
    batch_size: int = 128
    discount: float = 0.99

    # 学習率
    lr_init: float = 0.001
    lr_decay_rate: float = 0.1
    lr_decay_steps: int = 100_000

    # カテゴリ化する範囲
    v_min: int = -10
    v_max: int = 10

    # policyの温度パラメータのリスト
    policy_tau_schedule: List[dict] = None

    # td_steps: int = 5  # multisteps
    unroll_steps: int = 3  # unroll_steps

    # Root prior exploration noise.
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25

    # PUCT
    c_base: float = 19652
    c_init: float = 1.25

    # Priority Experience Replay
    capacity: int = 100_000
    memory_name: str = "ProportionalMemory"
    memory_warmup_size: int = 1000
    memory_alpha: float = 1.0
    memory_beta_initial: float = 1.0
    memory_beta_steps: int = 100_000

    # model
    input_image_block: kl.Layer = AlphaZeroImageBlock
    input_image_block_kwargs: dict = None
    dynamics_blocks: int = 15
    reward_dense_units: int = 0
    weight_decay: float = 0.0001

    # rescale
    enable_rescale: bool = True

    def set_atari_config(self):
        self.num_simulations = 50
        self.batch_size = 1024
        self.discount = 0.997
        self.lr_init = 0.05
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 350_000
        self.v_min = -300
        self.v_max = 300
        # self.td_steps = 10
        self.unroll_steps = 5
        self.policy_tau_schedule = [
            {"step": 0, "tau": 1.0},
            {"step": 500_000, "tau": 0.5},
            {"step": 750_000, "tau": 0.25},
        ]
        self.representation_block = MuZeroAtariBlock
        self.representation_block_kwargs = {"base_filters": 128}
        self.dynamics_blocks = 15
        self.weight_decay = 0.0001
        self.enable_rescale = True

    def __post_init__(self):
        if self.policy_tau_schedule is None:
            self.policy_tau_schedule = [
                {"step": 0, "tau": 1.0},
                {"step": 50_000, "tau": 0.5},
                {"step": 75_000, "tau": 0.25},
            ]
        if self.input_image_block_kwargs is None:
            self.input_image_block_kwargs = {}

    def set_processor(self) -> List[Processor]:
        return [
            ImageProcessor(
                image_type=EnvObservationType.GRAY_2ch,
                resize=(96, 96),
                enable_norm=True,
            )
        ]

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.CONTINUOUS

    @staticmethod
    def getName() -> str:
        return "MuZero"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.batch_size < self.memory_warmup_size
        assert self.policy_tau_schedule[0]["step"] == 0
        for tau in self.policy_tau_schedule:
            assert tau["tau"] >= 0
        assert self.v_min < self.v_max
        assert self.unroll_steps > 0


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
class RemoteMemory(PriorityExperienceReplay):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.init(
            self.config.memory_name,
            self.config.capacity,
            self.config.memory_alpha,
            self.config.memory_beta_initial,
            self.config.memory_beta_steps,
        )

        self.q_min = np.inf
        self.q_max = -np.inf

    def update_q(self, q_min, q_max):
        self.q_min = min(self.q_min, q_min)
        self.q_max = max(self.q_max, q_max)

    def get_q(self):
        return self.q_min, self.q_max


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _RepresentationNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        in_state, c, use_image_head = create_input_layer(
            config.observation_shape,
            config.env_observation_type,
        )
        assert use_image_head, "Input supports only image format."

        c = config.input_image_block(**config.input_image_block_kwargs)(c)
        self.model = keras.Model(in_state, c, name="RepresentationNetwork")

        # 重みを初期化
        dummy_state = np.zeros(shape=(1,) + config.observation_shape, dtype=np.float32)
        hidden_state = self(dummy_state)

        # 出力shape
        self.hidden_state_shape = hidden_state.shape[1:]

    def call(self, state):
        x = self.model(state)

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

        return x


class _DynamicsNetwork(keras.Model):
    def __init__(self, config: Config, input_shape):
        super().__init__()
        self.action_num = config.action_num
        v_num = config.v_max - config.v_min + 1
        h, w, ch = input_shape

        # hidden_state + action_space
        in_state = c = kl.Input(shape=(h, w, ch + self.action_num))

        # hidden_state
        c1 = AlphaZeroImageBlock(
            n_blocks=config.dynamics_blocks,
            filters=ch,
            l2=config.weight_decay,
        )(c)

        # reward
        c2 = kl.Conv2D(
            1,
            kernel_size=(1, 1),
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(config.weight_decay),
        )(c)
        c2 = kl.BatchNormalization()(c2)
        c2 = kl.ReLU()(c2)
        c2 = kl.Flatten()(c2)
        if config.reward_dense_units > 0:
            c2 = kl.Dense(
                config.reward_dense_units, activation="swish", kernel_regularizer=regularizers.l2(config.weight_decay)
            )(c2)
        reward = kl.Dense(v_num, activation="softmax", kernel_regularizer=regularizers.l2(config.weight_decay))(c2)

        self.model = keras.Model(in_state, [c1, reward], name="DynamicsNetwork")

        # 重みを初期化
        dummy1 = np.zeros(shape=(1,) + input_shape, dtype=np.float32)
        dummy2 = [1]
        v1, v2 = self(dummy1, dummy2)
        assert v1.shape == (1, h, w, ch)
        assert v2.shape == (1, v_num)

    def call(self, hidden_state, action):
        batch_size, h, w, _ = hidden_state.shape

        action_image = tf.one_hot(action, self.action_num)  # (batch, action)
        action_image = tf.repeat(action_image, repeats=h * w, axis=1)  # (batch, action * h * w)
        action_image = tf.reshape(action_image, (batch_size, self.action_num, h, w))  # (batch, action, h, w)
        action_image = tf.transpose(action_image, perm=[0, 2, 3, 1])  # (batch, h, w, action)

        in_state = tf.concat([hidden_state, action_image], axis=3)
        x, reward_category = self.model(in_state)

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

        v_num = config.v_max - config.v_min + 1

        in_layer = c = kl.Input(shape=input_shape)

        # --- policy
        c1 = kl.Conv2D(
            2,
            kernel_size=(1, 1),
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(config.weight_decay),
        )(c)
        c1 = kl.BatchNormalization()(c1)
        c1 = kl.ReLU()(c1)
        c1 = kl.Flatten()(c1)
        policy = kl.Dense(
            config.action_num, activation="softmax", kernel_regularizer=regularizers.l2(config.weight_decay)
        )(c1)

        # --- value
        c2 = kl.Conv2D(
            1,
            kernel_size=(1, 1),
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(config.weight_decay),
        )(c)
        c2 = kl.BatchNormalization()(c2)
        c2 = kl.ReLU()(c2)
        c2 = kl.Flatten()(c2)
        value = kl.Dense(v_num, activation="softmax", kernel_regularizer=regularizers.l2(config.weight_decay))(c2)

        self.model = keras.Model(in_layer, [policy, value], name="PredictionNetwork")

        # 重みを初期化
        dummy_state = np.zeros(shape=(1,) + input_shape, dtype=np.float32)
        policy, value = self(dummy_state)
        assert policy.shape == (1, config.action_num)
        assert value.shape == (1, v_num)

    def call(self, state):
        return self.model(state)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

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
        self.representation_network.model.summary(**kwargs)
        self.dynamics_network.model.summary(**kwargs)
        self.prediction_network.model.summary(**kwargs)

    # ------------------------

    def pred_PV(self, state, state_str):
        if state_str not in self.P:
            p, v_category = self.prediction_network(state)
            self.P[state_str] = p[0].numpy()
            self.V[state_str] = float_category_decode(v_category.numpy()[0], self.config.v_min, self.config.v_max)

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


class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        if compare_less_version(tf.__version__, "2.11.0"):
            self.optimizer = keras.optimizers.Adam()
        else:
            self.optimizer = keras.optimizers.legacy.Adam()

        self.train_count = 0

    def _cross_entropy_loss(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-6, y_pred)  # log(0)回避用
        loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1)
        return loss

    def get_train_count(self):
        return self.train_count

    def train(self):
        if self.remote_memory.length() < self.config.memory_warmup_size:
            return {}
        indices, batchs, weights = self.remote_memory.sample(self.train_count, self.config.batch_size)

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
            hidden_states = self.parameter.representation_network(states)
            p_pred, v_pred = self.parameter.prediction_network(hidden_states)

            # loss
            policy_loss = _scale_gradient(self._cross_entropy_loss(policies_list[0], p_pred), 1.0)
            value_loss = _scale_gradient(self._cross_entropy_loss(values_list[0], v_pred), 1.0)
            reward_loss = tf.constant([0] * self.config.batch_size, dtype=tf.float32)

            # --- unroll steps
            gradient_scale = 1 / self.config.unroll_steps
            for t in range(self.config.unroll_steps):
                # pred
                hidden_states, p_rewards = self.parameter.dynamics_network(hidden_states, actions_list[t])
                p_pred, v_pred = self.parameter.prediction_network(hidden_states)

                # loss
                policy_loss += _scale_gradient(self._cross_entropy_loss(policies_list[t + 1], p_pred), gradient_scale)
                value_loss += _scale_gradient(self._cross_entropy_loss(values_list[t + 1], v_pred), gradient_scale)
                reward_loss += _scale_gradient(self._cross_entropy_loss(rewards_list[t], p_rewards), gradient_scale)

                hidden_states = _scale_gradient(hidden_states, 0.5)

            loss = tf.reduce_mean((value_loss + policy_loss + reward_loss) * weights)

            # 各ネットワークの正則化項を加える
            loss += tf.reduce_sum(self.parameter.representation_network.losses)
            loss += tf.reduce_sum(self.parameter.prediction_network.losses)
            loss += tf.reduce_sum(self.parameter.dynamics_network.losses)

        priorities = value_loss.numpy()

        # lr
        lr = self.config.lr_init * self.config.lr_decay_rate ** (self.train_count / self.config.lr_decay_steps)
        self.optimizer.learning_rate = lr

        variables = [
            self.parameter.representation_network.trainable_variables,
            self.parameter.prediction_network.trainable_variables,
            self.parameter.dynamics_network.trainable_variables,
        ]
        grads = tape.gradient(loss, variables)
        for i in range(len(variables)):
            self.optimizer.apply_gradients(zip(grads[i], variables[i]))

        self.train_count += 1

        # memory update
        self.remote_memory.update(indices, batchs, priorities)

        # 学習したらキャッシュは削除
        self.parameter.reset_cache()

        # --- 正規化用Qを保存できるように送信(remote_memory -> trainer -> parameter)
        q_min, q_max = self.remote_memory.get_q()
        self.parameter.q_min = q_min
        self.parameter.q_max = q_max

        return {
            "value_loss": np.mean(value_loss),
            "policy_loss": np.mean(policy_loss),
            "reward_loss": np.mean(reward_loss),
            "loss": loss.numpy(),
            "lr": self.optimizer.learning_rate.numpy(),
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        self.policy_tau_schedule = {}
        for tau_list in self.config.policy_tau_schedule:
            self.policy_tau_schedule[tau_list["step"]] = tau_list["tau"]
        self.policy_tau = self.policy_tau_schedule[0]
        self.total_step = 0

        self._v_min = np.inf
        self._v_max = -np.inf

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        self.history = []

        self.N = {}  # 訪問回数(s,a)
        self.W = {}  # 累計報酬(s,a)
        self.Q = {}

        return {}

    def _init_state(self, state_str):
        if state_str not in self.N:
            self.N[state_str] = [0 for _ in range(self.config.action_num)]
            self.W[state_str] = [0 for _ in range(self.config.action_num)]
            self.Q[state_str] = [0 for _ in range(self.config.action_num)]

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        self.state = state
        self.invalid_actions = invalid_actions

        # --- シミュレーションしてpolicyを作成
        self.s0 = self.parameter.representation_network(state[np.newaxis, ...])
        self.s0_str = self.s0.ref()
        for _ in range(self.config.num_simulations):
            self._simulation(self.s0, self.s0_str, invalid_actions)

        # 正規化用Qを保存できるように送信(remote_memory -> trainer -> parameter)
        self.remote_memory.update_q(self.parameter.q_min, self.parameter.q_max)

        # V
        self.state_v = self.parameter.V[self.s0_str]

        # --- 確率に比例したアクションを選択
        if not self.training:
            self.policy_tau = 0  # 評価時は決定的に
        if self.policy_tau == 0:
            counts = np.asarray(self.N[self.s0_str])
            action = np.random.choice(np.where(counts == counts.max())[0])
        else:
            step_policy = np.array(
                [self.N[self.s0_str][a] ** (1 / self.policy_tau) for a in range(self.config.action_num)]
            )
            step_policy /= step_policy.sum()
            action = random_choice_by_probs(step_policy)

        # schedule check
        if self.total_step in self.policy_tau_schedule:
            self.policy_tau = self.policy_tau_schedule[self.total_step]

        # 学習用のpolicyはtau=1
        N = sum(self.N[self.s0_str])
        self.step_policy = [self.N[self.s0_str][a] / N for a in range(self.config.action_num)]

        self.action = int(action)
        return self.action, {}

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
        n_state, reward_category = self.parameter.dynamics_network(state, [action])
        n_state_str = n_state.ref()
        reward = float_category_decode(reward_category.numpy()[0], self.config.v_min, self.config.v_max)
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
                q = self.Q[state_str][a]

                # 過去観測したQ値で正規化(MinMax)
                if self.parameter.q_min < self.parameter.q_max:
                    q = (q - self.parameter.q_min) / (self.parameter.q_max - self.parameter.q_min)

                score = q + u
            scores[a] = score
        return scores

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ):
        self.total_step += 1

        if not self.training:
            return {}

        self.history.append(
            {
                "state": self.state,
                "action": self.action,
                "policy": self.step_policy,
                "reward": reward,
                "state_v": self.state_v,
            }
        )

        if done:
            zero_category = float_category_encode(0, self.config.v_min, self.config.v_max)

            # calc MC reward
            reward = 0
            for h in reversed(self.history):
                reward = h["reward"] + self.config.discount * reward
                h["discount_reward"] = reward

            # batch create
            for idx in range(len(self.history)):

                # --- policies
                policies = [
                    [1 / self.config.action_num] * self.config.action_num for _ in range(self.config.unroll_steps + 1)
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
                    values[i] = float_category_encode(v, self.config.v_min, self.config.v_max)
                priority /= self.config.unroll_steps + 1

                # --- actions
                actions = [random.randint(0, self.config.action_num - 1) for _ in range(self.config.unroll_steps)]
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
                    rewards[i] = float_category_encode(r, self.config.v_min, self.config.v_max)

                self.remote_memory.add(
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

    def render_terminal(self, env, worker, **kwargs) -> None:
        self._init_state(self.s0_str)
        self.parameter.pred_PV(self.s0, self.s0_str)
        puct = self._calc_puct(self.s0_str, self.invalid_actions, False)
        maxa = self.action

        v = self.parameter.V[self.s0_str]
        if self.config.enable_rescale:
            v = inverse_rescaling(v)

        print(f"V_net: {v:.5f}")

        def _render_sub(a: int) -> str:
            q = self.Q[self.s0_str][a]
            c = self.N[self.s0_str][a]
            p = self.step_policy[a]

            n_s, reward_category = self.parameter.dynamics_network(self.s0, [a])
            reward = float_category_decode(reward_category.numpy()[0], self.config.v_min, self.config.v_max)
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

        render_discrete_action(self.invalid_actions, maxa, env, _render_sub)
