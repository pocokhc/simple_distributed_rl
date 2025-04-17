from dataclasses import dataclass, field
from typing import Any, List, cast

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import SpaceTypes
from srl.base.exception import UndefinedError
from srl.base.rl.algorithms.base_ppo import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import RLProcessor
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.memories.replay_buffer import ReplayBufferConfig, RLReplayBuffer
from srl.rl.models.config.input_image_block import InputImageBlockConfig
from srl.rl.models.config.input_value_block import InputValueBlockConfig
from srl.rl.models.config.mlp_block import MLPBlockConfig
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig
from srl.rl.tf import helper as helper_tf
from srl.rl.tf.distributions.categorical_dist_block import CategoricalDistBlock
from srl.rl.tf.distributions.normal_dist_block import NormalDistBlock
from srl.rl.tf.model import KerasModelAddedSummary

kl = keras.layers

"""
Paper
https://arxiv.org/abs/1812.05905

DDPG
    Replay buffer       : o
    Target Network(soft): o
    Target Network(hard): o
    Add action noise    : x
TD3
    Clipped Double Q learning : o
    Target Policy Smoothing   : x
    Delayed Policy Update     : x
SAC
    Squashed Gaussian Policy: o
"""


@dataclass
class Config(RLConfig):
    #: <:ref:`InputValueBlockConfig`>
    input_value_block: InputValueBlockConfig = field(default_factory=lambda: InputValueBlockConfig())
    #: <:ref:`InputImageBlockConfig`>
    input_image_block: InputImageBlockConfig = field(default_factory=lambda: InputImageBlockConfig())
    #: <:ref:`MLPBlockConfig`> policy layer
    policy_hidden_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig().set((64, 64, 64)))
    #: <:ref:`MLPBlockConfig`>
    q_hidden_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig().set((128, 128, 128)))

    #: Batch size
    batch_size: int = 32
    #: <:ref:`ReplayBufferConfig`>
    memory: ReplayBufferConfig = field(default_factory=lambda: ReplayBufferConfig(capacity=1000))

    #: discount
    discount: float = 0.9
    #: policy learning rate
    lr_policy: float = 0.001
    #: <:ref:`LRSchedulerConfig`>
    lr_policy_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())
    #: q learning rate
    lr_q: float = 0.001
    #: <:ref:`LRSchedulerConfig`>
    lr_q_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())
    #: alpha learning rate
    lr_alpha: float = 0.001
    #: <:ref:`LRSchedulerConfig`>
    lr_alpha_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())
    #: soft_target_update_tau
    soft_target_update_tau: float = 0.02
    #: hard_target_update_interval
    hard_target_update_interval: int = 100
    #: actionが連続値の時、正規分布をtanhで-1～1に丸めるか
    enable_normal_squashed: bool = True

    #: entropy alphaを自動調整するか
    entropy_alpha_auto_scale: bool = True
    #: entropy alphaの初期値
    entropy_alpha: float = 0.2
    #: Q値の計算からエントロピーボーナスを除外します
    entropy_bonus_exclude_q: float = False

    #: 勾配爆発の対策, 平均、分散、ランダムアクションで大きい値を出さないようにclipする
    enable_stable_gradients: bool = True
    #: enable_stable_gradients状態での標準偏差のclip
    stable_gradients_scale_range: tuple = (1e-10, 10)

    def get_name(self) -> str:
        return "SAC"

    def get_framework(self) -> str:
        return "tensorflow"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if prev_observation_space.is_image():
            return self.input_image_block.get_processors()
        return []


register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


class Memory(RLReplayBuffer):
    pass


class PolicyNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        if config.observation_space.is_image():
            self.in_block = config.input_image_block.create_tf_block(config.observation_space)
        else:
            self.in_block = config.input_value_block.create_tf_block(config.observation_space)

        self.hidden_block = config.policy_hidden_block.create_tf_block()

        # out
        if isinstance(config.action_space, DiscreteSpace):
            self.policy_dist_block = CategoricalDistBlock(config.action_space.n)
        elif isinstance(config.action_space, ArrayContinuousSpace):
            self.policy_dist_block = NormalDistBlock(
                cast(ArrayContinuousSpace, config.action_space).size,
                enable_squashed=self.config.enable_normal_squashed,
                enable_stable_gradients=self.config.enable_stable_gradients,
                stable_gradients_scale_range=self.config.stable_gradients_scale_range,
            )
        else:
            raise UndefinedError(self.config.action_space)

        # build
        self(self.in_block.create_dummy_data(config.get_dtype("np")))

    def call(self, x, training=False) -> Any:
        x = self.in_block(x, training=training)
        x = self.hidden_block(x, training=training)
        return self.policy_dist_block(x)

    @tf.function
    def compute_train_loss(self, state, q1_model, q2_model, alpha):
        p_dist = self(state, training=True)

        if self.config.action_space.stype == SpaceTypes.DISCRETE:
            action = p_dist.rsample()
            logpi = p_dist.log_probs()
            entropy = -tf.reduce_sum(tf.exp(logpi) * logpi, axis=1, keepdims=True)
        elif self.config.action_space.stype == SpaceTypes.CONTINUOUS:
            action, logpi = p_dist.rsample_logprob()
            entropy = -logpi
        else:
            raise UndefinedError(self.config.action_space.stype)

        # Q値を出力、小さいほうを使う
        q1 = q1_model([state, action])
        q2 = q2_model([state, action])
        q_min = tf.minimum(q1, q2)

        policy_loss = q_min + (alpha * entropy)
        policy_loss = -tf.reduce_mean(policy_loss)
        policy_loss += tf.reduce_sum(self.losses)  # 正則化項
        return policy_loss, logpi


class QNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config):
        super().__init__()

        if config.observation_space.is_image():
            self.in_block = config.input_image_block.create_tf_block(config.observation_space)
        else:
            self.in_block = config.input_value_block.create_tf_block(config.observation_space)

        self.q_block = config.q_hidden_block.create_tf_block()
        self.q_out_layer = kl.Dense(1)

        # build
        if isinstance(config.action_space, DiscreteSpace):
            self._in_shape2 = (config.action_space.n,)
        else:
            self._in_shape2 = (config.action_space.size,)
        self(
            [
                self.in_block.create_dummy_data(config.get_dtype("np")),
                np.zeros((1,) + self._in_shape2),
            ]
        )

    def call(self, x, training=False):
        state = x[0]
        onehot_action = x[1]

        state = self.in_block(state, training=training)
        x = tf.concat([state, onehot_action], axis=1)

        x = self.q_block(x, training=training)
        x = self.q_out_layer(x, training=training)
        return x

    @tf.function
    def compute_train_loss(self, state, onehot_action, target_q):
        q = self([state, onehot_action], training=True)
        loss = tf.reduce_mean(tf.square(target_q - q))
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss


class Parameter(RLParameter):
    def setup(self):
        self.policy = PolicyNetwork(self.config)
        self.q1_online = QNetwork(self.config)
        self.q1_target = QNetwork(self.config)
        self.q1_target.set_weights(self.q1_online.get_weights())
        self.q2_online = QNetwork(self.config)
        self.q2_target = QNetwork(self.config)
        self.q2_target.set_weights(self.q2_online.get_weights())

        # エントロピーα自動調整用
        self.log_alpha = tf.Variable(np.log(self.config.entropy_alpha), dtype=tf.float32)
        self.load_log_alpha = False

    def call_restore(self, data: Any, **kwargs) -> None:
        self.policy.set_weights(data[0])
        self.q1_online.set_weights(data[1])
        self.q1_target.set_weights(data[1])
        self.q2_online.set_weights(data[2])
        self.q2_target.set_weights(data[2])
        self.log_alpha = data[3]
        self.load_log_alpha = False

    def call_backup(self, **kwargs) -> Any:
        return [
            self.policy.get_weights(),
            self.q1_online.get_weights(),
            self.q2_online.get_weights(),
            self.log_alpha,
        ]

    def summary(self, **kwargs):
        self.policy.summary(**kwargs)
        self.q1_online.summary(**kwargs)


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.q1_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr_q_scheduler.apply_tf_scheduler(self.config.lr_q))
        self.q2_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr_q_scheduler.apply_tf_scheduler(self.config.lr_q))
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr_policy_scheduler.apply_tf_scheduler(self.config.lr_policy))
        self.alpha_optimizer = None

        # エントロピーαの目標値、-1*アクション数が良いらしい
        if self.config.action_space.stype == SpaceTypes.DISCRETE:
            n = self.config.action_space.n
        else:
            n = self.config.action_space.size
        self.target_entropy = -1 * n

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return

        (
            states,
            actions,
            n_states,
            rewards,
            dones,
        ) = zip(*batches)
        states = np.asarray(states)
        actions = np.asarray(actions)
        n_states = np.asarray(n_states)
        dones = np.asarray(dones, dtype=np.float32)[..., np.newaxis]
        rewards = np.asarray(rewards, dtype=np.float32)[..., np.newaxis]

        if not self.parameter.load_log_alpha:
            # restore時に再度作り直す必要あり
            self.alpha_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr_alpha_scheduler.apply_tf_scheduler(self.config.lr_alpha))
            self.parameter.load_log_alpha = True

        # 方策エントロピーの反映率αを計算
        alpha = np.exp(self.parameter.log_alpha)

        # ポリシーより次の状態のアクションを取得し、次の状態のアクションlogpiを取得
        n_p_dist = self.parameter.policy(n_states)
        if self.config.action_space.stype == SpaceTypes.DISCRETE:
            n_action = n_p_dist.rsample()
            n_logpi = n_p_dist.log_prob(n_action)
            entropy = -n_logpi
        elif self.config.action_space.stype == SpaceTypes.CONTINUOUS:
            n_action, n_logpi = n_p_dist.rsample_logprob()
            entropy = -n_logpi
        else:
            raise UndefinedError(self.config.action_space.stype)

        # 2つのQ値から小さいほうを採用(Clipped Double Q learning)
        n_q1 = self.parameter.q1_target([n_states, n_action])
        n_q2 = self.parameter.q2_target([n_states, n_action])
        n_qval = tf.minimum(n_q1, n_q2)
        if self.config.entropy_bonus_exclude_q:
            target_q = rewards + (1 - dones) * self.config.discount * n_qval
        else:
            target_q = rewards + (1 - dones) * self.config.discount * (n_qval + alpha * entropy)

        # --- Qモデルの学習
        # 一緒に学習すると-と+で釣り合う場合がある
        self.parameter.q1_online.trainable = True
        self.parameter.q2_online.trainable = True
        with tf.GradientTape() as tape:
            q1_loss = self.parameter.q1_online.compute_train_loss(states, actions, target_q)
        grads = tape.gradient(q1_loss, self.parameter.q1_online.trainable_variables)
        self.q1_optimizer.apply_gradients(zip(grads, self.parameter.q1_online.trainable_variables))
        self.info["q1_loss"] = q1_loss.numpy()

        with tf.GradientTape() as tape:
            q2_loss = self.parameter.q2_online.compute_train_loss(states, actions, target_q)
        grads = tape.gradient(q2_loss, self.parameter.q2_online.trainable_variables)
        self.q2_optimizer.apply_gradients(zip(grads, self.parameter.q2_online.trainable_variables))
        self.info["q2_loss"] = q2_loss.numpy()

        # --- ポリシーの学習
        self.parameter.q1_online.trainable = False
        self.parameter.q2_online.trainable = False
        with tf.GradientTape() as tape:
            policy_loss, logpi = self.parameter.policy.compute_train_loss(
                states,
                self.parameter.q1_online,
                self.parameter.q2_online,
                alpha,
            )
        grads = tape.gradient(policy_loss, self.parameter.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(grads, self.parameter.policy.trainable_variables))
        self.info["policy_loss"] = policy_loss.numpy()

        # --- 方策エントロピーαの自動調整
        if self.config.entropy_alpha_auto_scale:
            with tf.GradientTape() as tape:
                entropy_diff = logpi + self.target_entropy
                log_alpha_loss = tf.reduce_mean(-tf.exp(self.parameter.log_alpha) * entropy_diff)
            grad = tape.gradient(log_alpha_loss, self.parameter.log_alpha)
            self.alpha_optimizer.apply_gradients([(grad, self.parameter.log_alpha)])
            self.info["alpha_loss"] = log_alpha_loss.numpy()
            self.info["alpha"] = alpha

        # --- target sync
        if self.train_count % self.config.hard_target_update_interval == 0:
            self.parameter.q1_target.set_weights(self.parameter.q1_online.get_weights())
            self.parameter.q2_target.set_weights(self.parameter.q2_online.get_weights())
        else:
            helper_tf.model_soft_sync(self.parameter.q1_target, self.parameter.q1_online, self.config.soft_target_update_tau)
            helper_tf.model_soft_sync(self.parameter.q2_target, self.parameter.q2_online, self.config.soft_target_update_tau)

        self.train_count += 1


class Worker(RLWorker[Config, Parameter, Memory]):
    def policy(self, worker) -> Any:
        p_dist = self.parameter.policy(worker.state[np.newaxis, ...])
        if isinstance(self.config.action_space, DiscreteSpace):
            self.action = p_dist.sample(onehot=True).numpy()[0]
            env_action = int(np.argmax(self.action))
            if self.rendering:
                self.probs = p_dist.probs().numpy()[0]

            # --- debug
            # env_action = self.sample_action()
            # self.action = np.identity(self.config.action_space.n, dtype=np.float32)[env_action]

        elif isinstance(self.config.action_space, ArrayContinuousSpace):
            act_space = self.config.action_space
            self.action, env_action = p_dist.policy(act_space.low, act_space.high, self.training)
            self.action = self.action.numpy()[0]
            env_action = env_action.numpy()[0].tolist()
        else:
            raise UndefinedError(self.config.action_space)

        return env_action

    def on_step(self, worker):
        if not self.training:
            return

        """
        [
            state,
            action,
            n_state,
            reward,
            done,
        ]
        """
        self.memory.add(
            [
                worker.state,
                self.action,
                worker.state,
                worker.reward,
                worker.done,
            ]
        )

    def render_terminal(self, worker, **kwargs) -> None:
        # policy -> render -> env.step
        state = worker.state[np.newaxis, ...]
        if isinstance(self.config.action_space, DiscreteSpace):
            maxa = np.argmax(self.probs)

            def _render_sub(a: int) -> str:
                onehot_a = np.identity(self.config.action_space.n, dtype=np.float32)[a][np.newaxis, ...]
                q1 = self.parameter.q1_online([state, onehot_a])
                q2 = self.parameter.q2_online([state, onehot_a])
                q1 = q1.numpy()[0][0]
                q2 = q2.numpy()[0][0]

                s = f"{self.probs[a] * 100:5.1f}%, q1 {q1:.5f}, q2 {q2:.5f} "
                return s

            worker.print_discrete_action_info(int(maxa), _render_sub)

        elif isinstance(self.config.action_space, ArrayContinuousSpace):
            dist = self.parameter.policy(state)
            print(f"action: {self.action}")
            print(f"mean  : {dist.mean().numpy()[0][0]}")
            print(f"stddev: {dist.stddev().numpy()[0][0]}")
            q1 = self.parameter.q1_online([state, self.action[np.newaxis, ...]])
            q2 = self.parameter.q2_online([state, self.action[np.newaxis, ...]])
            q1 = q1.numpy()[0][0]
            q2 = q2.numpy()[0][0]
            print(f"q1 {q1:8.5f}")
            print(f"q2 {q2:8.5f}")

        else:
            raise UndefinedError(self.config.action_space)
