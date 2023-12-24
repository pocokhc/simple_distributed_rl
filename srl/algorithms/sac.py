from dataclasses import dataclass, field
from typing import Any, List, Tuple, cast

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import EnvObservationTypes, RLBaseTypes, RLTypes
from srl.base.exception import UndefinedError
from srl.base.rl.base import RLParameter, RLTrainer, RLWorker
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.rl.registration import register
from srl.base.rl.worker_run import WorkerRun
from srl.rl.functions.common import render_discrete_action
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, ExperienceReplayBufferConfig
from srl.rl.models.image_block import ImageBlockConfig
from srl.rl.models.mlp_block import MLPBlockConfig
from srl.rl.models.tf.distributions.categorical_gumbel_dist_block import CategoricalGumbelDistBlock
from srl.rl.models.tf.distributions.normal_dist_block import NormalDistBlock
from srl.rl.models.tf.input_block import InputImageBlock
from srl.rl.processors.image_processor import ImageProcessor
from srl.rl.schedulers.scheduler import SchedulerConfig
from srl.utils.common import compare_less_version

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


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig, ExperienceReplayBufferConfig):
    # --- model
    #: <:ref:`ImageBlock`> This layer is only used when the input is an image.
    image_block: ImageBlockConfig = field(init=False, default_factory=lambda: ImageBlockConfig())
    #: <:ref:`MLPBlock`> policy layer
    policy_hidden_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())
    #: <:ref:`MLPBlock`>
    q_hidden_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())

    #: discount
    discount: float = 0.9
    #: policy learning rate
    lr_policy: float = 0.001  # type: ignore , type OK
    #: q learning rate
    lr_q: float = 0.001  # type: ignore , type OK
    #: alpha learning rate
    lr_alpha: float = 0.001  # type: ignore , type OK
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
    stable_gradients_stddev_range: tuple = (1e-10, 10)

    def __post_init__(self):
        super().__post_init__()

        self.memory.capacity = 1000
        self.lr_policy: SchedulerConfig = SchedulerConfig(cast(float, self.lr_policy))
        self.lr_q: SchedulerConfig = SchedulerConfig(cast(float, self.lr_q))
        self.lr_alpha: SchedulerConfig = SchedulerConfig(cast(float, self.lr_alpha))
        self.policy_hidden_block.set_mlp((64, 64, 64))
        self.q_hidden_block.set_mlp((128, 128, 128))

    @property
    def base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.ANY

    @property
    def base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.CONTINUOUS

    def get_use_framework(self) -> str:
        return "tensorflow"

    def set_processor(self) -> List[Processor]:
        return [
            ImageProcessor(
                image_type=EnvObservationTypes.GRAY_2ch,
                resize=(84, 84),
                enable_norm=True,
            )
        ]

    def getName(self) -> str:
        return "SAC"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_memory()


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
class _PolicyNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # input
        self.in_img_block = None
        if config.observation_type == RLTypes.IMAGE:
            self.in_img_block = InputImageBlock(config.observation_shape, config.env_observation_type)
            self.img_block = config.image_block.create_block_tf()
        self.flat_layer = kl.Flatten()

        # layers
        self.hidden_block = config.policy_hidden_block.create_block_tf()
        if self.config.action_type == RLTypes.DISCRETE:
            self.policy_dist_block = CategoricalGumbelDistBlock(config.action_num)
        elif self.config.action_type == RLTypes.CONTINUOUS:
            self.policy_dist_block = NormalDistBlock(
                config.action_num,
                enable_squashed=self.config.enable_normal_squashed,
                enable_stable_gradients=self.config.enable_stable_gradients,
                stable_gradients_stddev_range=self.config.stable_gradients_stddev_range,
            )
        else:
            raise UndefinedError(self.config.action_type)

        # build
        self._in_shape = config.observation_shape
        self.build((None,) + self._in_shape)

    def call(self, x, training=False) -> Any:
        if self.in_img_block is not None:
            x = self.in_img_block(x, training)
            x = self.img_block(x, training)
        x = self.flat_layer(x)
        x = self.hidden_block(x, training=training)
        return self.policy_dist_block(x)

    def policy(self, state):
        return self.policy_dist_block.get_dist(self(state, training=False))

    @tf.function
    def compute_train_loss(self, state, q1_model, q2_model, alpha):
        p_dist = self.policy_dist_block.get_grad_dist(self(state, training=True))

        if self.config.action_type == RLTypes.DISCRETE:
            action = p_dist.sample()
            logpi = p_dist.log_probs()
            H = -tf.reduce_sum(tf.exp(logpi) * logpi, axis=1, keepdims=True)
        elif self.config.action_type == RLTypes.CONTINUOUS:
            action = p_dist.sample()
            logpi = p_dist.log_prob(p_dist.y_org)
            H = -logpi
        else:
            raise UndefinedError(self.config.action_type)

        # Q値を出力、小さいほうを使う
        q1 = q1_model([state, action])
        q2 = q2_model([state, action])
        q_min = tf.minimum(q1, q2)

        policy_loss = q_min + (alpha * H)
        policy_loss = -tf.reduce_mean(policy_loss)
        policy_loss += tf.reduce_sum(self.losses)  # 正則化項
        return policy_loss, logpi

    def summary(self, name: str = "", **kwargs):
        if self.in_img_block is not None:
            self.in_img_block.init_model_graph()
            self.img_block.init_model_graph()
        self.hidden_block.init_model_graph()

        x = kl.Input(shape=self._in_shape)
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x), name=name)
        model.summary(**kwargs)


class _QNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        # input
        self.in_img_block = None
        if config.observation_type == RLTypes.IMAGE:
            self.in_img_block = InputImageBlock(config.observation_shape, config.env_observation_type)
            self.img_block = config.image_block.create_block_tf()
        self.flat_layer = kl.Flatten()

        self.q_block = config.q_hidden_block.create_block_tf()
        self.q_out_layer = kl.Dense(1)

        # build
        self._in_shape1 = config.observation_shape
        self._in_shape2 = (config.action_num,)
        self.build([(None,) + self._in_shape1, (None,) + self._in_shape2])

    def call(self, x, training=False):
        state = x[0]
        onehot_action = x[1]

        if self.in_img_block is not None:
            state = self.in_img_block(state, training)
            state = self.img_block(state, training)
        state = self.flat_layer(state)
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

    def summary(self, name: str = "", **kwargs):
        if self.in_img_block is not None:
            self.in_img_block.init_model_graph()
            self.img_block.init_model_graph()
        self.q_block.init_model_graph()

        x = [kl.Input(shape=self._in_shape1), kl.Input(shape=self._in_shape2)]
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

        self.policy = _PolicyNetwork(self.config)
        self.q1_online = _QNetwork(self.config)
        self.q1_target = _QNetwork(self.config)
        self.q1_target.set_weights(self.q1_online.get_weights())
        self.q2_online = _QNetwork(self.config)
        self.q2_target = _QNetwork(self.config)
        self.q2_target.set_weights(self.q2_online.get_weights())

        # エントロピーα自動調整用
        self.log_alpha = tf.Variable(np.log(self.config.entropy_alpha), dtype=tf.float32)

    def call_restore(self, data: Any, **kwargs) -> None:
        self.policy.set_weights(data[0])
        self.q1_online.set_weights(data[1])
        self.q1_target.set_weights(data[1])
        self.q2_online.set_weights(data[2])
        self.q2_target.set_weights(data[2])
        self.log_alpha = data[3]

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


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_q_sch = self.config.lr_q.create_schedulers()
        self.lr_policy_sch = self.config.lr_policy.create_schedulers()
        self.lr_alpha_sch = self.config.lr_alpha.create_schedulers()

        if compare_less_version(tf.__version__, "2.11.0"):
            self.q1_optimizer = keras.optimizers.Adam(learning_rate=self.lr_q_sch.get_rate())
            self.q2_optimizer = keras.optimizers.Adam(learning_rate=self.lr_q_sch.get_rate())
            self.policy_optimizer = keras.optimizers.Adam(learning_rate=self.lr_policy_sch.get_rate())
            self.alpha_optimizer = keras.optimizers.Adam(learning_rate=self.lr_alpha_sch.get_rate())
        else:
            self.q1_optimizer = keras.optimizers.legacy.Adam(learning_rate=self.lr_q_sch.get_rate())
            self.q2_optimizer = keras.optimizers.legacy.Adam(learning_rate=self.lr_q_sch.get_rate())
            self.policy_optimizer = keras.optimizers.legacy.Adam(learning_rate=self.lr_policy_sch.get_rate())
            self.alpha_optimizer = keras.optimizers.legacy.Adam(learning_rate=self.lr_alpha_sch.get_rate())

        # エントロピーαの目標値、-1*アクション数が良いらしい
        self.target_entropy = -1 * self.config.action_num

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        batchs = self.memory.sample(self.batch_size, self.train_count)

        states = []
        actions = []
        n_states = []
        rewards = []
        dones = []
        for b in batchs:
            states.append(b["state"])
            actions.append(b["action"])
            n_states.append(b["next_state"])
            rewards.append(b["reward"])
            dones.append(b["done"])
        states = np.asarray(states)
        n_states = np.asarray(n_states)
        actions = np.asarray(actions)
        dones = np.asarray(dones, dtype=np.float32)[..., np.newaxis]
        rewards = np.asarray(rewards, dtype=np.float32)[..., np.newaxis]

        # 方策エントロピーの反映率αを計算
        alpha = np.exp(self.parameter.log_alpha)

        # ポリシーより次の状態のアクションを取得し、次の状態のアクションlogpiを取得
        n_p_dist = self.parameter.policy.policy(n_states)
        if self.config.action_type == RLTypes.DISCRETE:
            n_action = n_p_dist.sample(onehot=True)
            n_logpi = n_p_dist.log_prob(n_action)
            H = -n_logpi
        elif self.config.action_type == RLTypes.CONTINUOUS:
            n_action = n_p_dist.sample()
            # Squashed Gaussian PolicyはSquash前のアクションを渡す必要あり
            n_logpi = n_p_dist.log_prob(n_p_dist.y_org)
            H = -n_logpi
        else:
            raise UndefinedError(self.config.action_type)

        # 2つのQ値から小さいほうを採用(Clipped Double Q learning)
        n_q1 = self.parameter.q1_target([n_states, n_action])
        n_q2 = self.parameter.q2_target([n_states, n_action])
        n_qval = tf.minimum(n_q1, n_q2)
        if self.config.entropy_bonus_exclude_q:
            target_q = rewards + (1 - dones) * self.config.discount * n_qval
        else:
            target_q = rewards + (1 - dones) * self.config.discount * (n_qval + alpha * H)

        # --- Qモデルの学習
        # 一緒に学習すると-と+で釣り合う場合がある
        self.parameter.q1_online.trainable = True
        self.parameter.q2_online.trainable = True
        with tf.GradientTape() as tape:
            q1_loss = self.parameter.q1_online.compute_train_loss(states, actions, target_q)
        grads = tape.gradient(q1_loss, self.parameter.q1_online.trainable_variables)
        self.q1_optimizer.apply_gradients(zip(grads, self.parameter.q1_online.trainable_variables))
        self.train_info["q1_loss"] = q1_loss.numpy()

        with tf.GradientTape() as tape:
            q2_loss = self.parameter.q2_online.compute_train_loss(states, actions, target_q)
        grads = tape.gradient(q2_loss, self.parameter.q2_online.trainable_variables)
        self.q2_optimizer.apply_gradients(zip(grads, self.parameter.q2_online.trainable_variables))
        self.train_info["q2_loss"] = q2_loss.numpy()

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
        self.train_info["policy_loss"] = policy_loss.numpy()

        # --- 方策エントロピーαの自動調整
        if self.config.entropy_alpha_auto_scale:
            with tf.GradientTape() as tape:
                entropy_diff = logpi + self.target_entropy
                log_alpha_loss = tf.reduce_mean(-tf.exp(self.parameter.log_alpha) * entropy_diff)
            grad = tape.gradient(log_alpha_loss, self.parameter.log_alpha)
            self.alpha_optimizer.apply_gradients([(grad, self.parameter.log_alpha)])
            self.train_info["alpha_loss"] = log_alpha_loss.numpy()
            self.train_info["alpha"] = alpha

        # --- soft target update
        self.parameter.q1_target.set_weights(
            (1 - self.config.soft_target_update_tau) * np.array(self.parameter.q1_target.get_weights(), dtype=object)
            + (self.config.soft_target_update_tau) * np.array(self.parameter.q1_online.get_weights(), dtype=object)
        )
        self.parameter.q2_target.set_weights(
            (1 - self.config.soft_target_update_tau) * np.array(self.parameter.q2_target.get_weights(), dtype=object)
            + (self.config.soft_target_update_tau) * np.array(self.parameter.q2_online.get_weights(), dtype=object)
        )

        # --- hard target sync
        if self.train_count % self.config.hard_target_update_interval == 0:
            self.parameter.q1_target.set_weights(self.parameter.q1_online.get_weights())
            self.parameter.q2_target.set_weights(self.parameter.q2_online.get_weights())

        # lr_schedule
        if self.lr_q_sch.update(self.train_count):
            self.q1_optimizer.learning_rate = self.lr_q_sch.get_rate()
            self.q2_optimizer.learning_rate = self.lr_q_sch.get_rate()
        if self.lr_policy_sch.update(self.train_count):
            self.policy_optimizer.learning_rate = self.lr_policy_sch.get_rate()
        if self.lr_alpha_sch.update(self.train_count):
            self.alpha_optimizer.learning_rate = self.lr_alpha_sch.get_rate()

        self.train_count += 1


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

    def on_reset(self, worker: WorkerRun) -> dict:
        return {}

    def policy(self, worker: WorkerRun) -> Tuple[Any, dict]:
        self.state = worker.state

        p_dist = self.parameter.policy.policy(self.state[np.newaxis, ...])
        if self.config.action_type == RLTypes.DISCRETE:  # int
            self.action = p_dist.sample(onehot=True).numpy()[0]
            env_action = int(np.argmax(self.action))
            if self.rendering:
                self.probs = p_dist.probs().numpy()[0]

            # --- debug
            # env_action = self.sample_action()
            # self.action = np.identity(self.config.action_num, dtype=np.float32)[env_action]

        elif self.config.action_type == RLTypes.CONTINUOUS:  # float,list[float]
            if self.training:
                self.action = p_dist.sample().numpy()[0]
            else:
                self.action = p_dist.mean().numpy()[0]

            if self.config.enable_normal_squashed:
                # Squashed Gaussian Policy (-1, 1) -> (action range)
                env_action = (self.action + 1) / 2
                env_action = self.config.action_low + env_action * (self.config.action_high - self.config.action_low)
            else:
                env_action = np.clip(self.action, self.config.action_low, self.config.action_high)
            env_action = env_action.tolist()

            if self.rendering:
                self.mean = p_dist.mean().numpy()[0]
                self.stddev = p_dist.stddev().numpy()[0]

        else:
            raise UndefinedError(self.config.action_type)

        return env_action, {}

    def on_step(self, worker: WorkerRun) -> dict:
        if not self.training:
            return {}

        batch = {
            "state": self.state,
            "action": self.action,
            "next_state": worker.state,
            "reward": worker.reward,
            "done": worker.done,
        }
        self.memory.add(batch)

        return {}

    def render_terminal(self, worker, **kwargs) -> None:
        if self.config.action_type == RLTypes.DISCRETE:
            maxa = np.argmax(self.probs)

            def _render_sub(a: int) -> str:
                onehot_a = np.identity(self.config.action_num, dtype=np.float32)[a][np.newaxis, ...]
                q1 = self.parameter.q1_online([self.state[np.newaxis, ...], onehot_a])
                q2 = self.parameter.q2_online([self.state[np.newaxis, ...], onehot_a])
                q1 = q1.numpy()[0][0]
                q2 = q2.numpy()[0][0]

                s = f"{self.probs[a] * 100:5.1f}%, q1 {q1:.5f}, q2 {q2:.5f} "
                return s

            render_discrete_action(maxa, worker.env, self.config, _render_sub)

        elif self.config.action_type == RLTypes.CONTINUOUS:
            q1 = self.parameter.q1_online([self.state[np.newaxis, ...], self.action[np.newaxis, ...]])
            q2 = self.parameter.q2_online([self.state[np.newaxis, ...], self.action[np.newaxis, ...]])
            q1 = q1.numpy()[0][0]
            q2 = q2.numpy()[0][0]
            print(f"q1 {q1:8.5f}")
            print(f"q2 {q2:8.5f}")
            print(f"action: {self.action}")
            print(f"mean  : {self.mean}")
            print(f"stddev: {self.stddev}")

        else:
            raise UndefinedError(self.config.action_type)
