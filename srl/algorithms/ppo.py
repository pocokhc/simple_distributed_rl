from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import EnvObservationTypes, RLTypes
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.rl.registration import register
from srl.base.rl.worker_rl import RLWorker
from srl.base.rl.worker_run import WorkerRun
from srl.rl.functions.common import render_discrete_action
from srl.rl.functions.common_tf import compute_kl_divergence, compute_kl_divergence_normal, compute_logprob
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, ExperienceReplayBufferConfig
from srl.rl.models.image_block import ImageBlockConfig
from srl.rl.models.mlp_block import MLPBlockConfig
from srl.rl.models.tf.input_block import InputBlock
from srl.rl.processors.image_processor import ImageProcessor

kl = keras.layers

"""
Paper
https://arxiv.org/abs/1707.06347
https://arxiv.org/abs/2005.12729

Clipped Surrogate Objective : o
Adaptive KL Penalty         : o
GAE                         : o
Other
  Value Clipping : o
  Reward scaling : o
  Orthogonal initialization and layer scaling: x
  Adam learning rate annealing : o
  Reward Clipping              : o
  Observation Normalization    : o
  Observation Clipping         : o
  Hyperbolic tan activations   : x
  Global Gradient Clipping     : o
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig, ExperienceReplayBufferConfig):
    # model
    image_block: ImageBlockConfig = field(init=False, default_factory=lambda: ImageBlockConfig())
    hidden_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())
    value_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())
    policy_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())

    experience_collection_method: str = "MC"  # "" or "MC" or "GAE"
    gae_discount: float = 0.9  # GAEの割引率

    baseline_type: str = "ave"  # "" or "ave" or "std" or "normal" or "advantage"
    surrogate_type: str = "clip"  # "" or "clip" or "kl"
    clip_range: float = 0.2  # 状態価値のクリップ範囲
    adaptive_kl_target: float = 0.01  # Adaptive KLペナルティ内の定数

    batch_size: int = 32
    memory_warmup_size: int = 1000
    discount: float = 0.9  # 割引率
    optimizer_initial_lr: float = 0.02  # 初期学習率
    optimizer_final_lr: float = 0.01  # 終了学習率
    optimizer_lr_step: float = 200 * 10  # 終了学習率になるまでの更新回数
    value_loss_weight: float = 1.0  # 状態価値の反映率
    entropy_weight: float = 0.1  # エントロピーの反映率

    enable_state_normalized: bool = True  # 状態の正規化
    enable_value_clip: float = True  # 価値関数もclipするか
    global_gradient_clip_norm: float = 0.5  # 勾配のL2におけるclip値(0で無効)

    enable_action_normalization: bool = True  # アクションの正規化
    state_clip: Optional[Tuple[float, float]] = None  # 状態のclip(Noneで無効、(-10,10)で指定)
    reward_clip: Optional[Tuple[float, float]] = None  # 報酬のclip(Noneで無効、(-10,10)で指定)

    enable_stable_gradients: bool = True  # 勾配爆発の対策
    """ 勾配爆発の対策, 平均、分散、ランダムアクションで大きい値を出さないようにclipする """
    stable_gradients_max_stddev: float = 2

    def __post_init__(self):
        super().__post_init__()

        self.memory.capacity = 2000
        self.hidden_block.set_mlp((64, 64))
        self.value_block.set_mlp((64,))
        self.policy_block.set_mlp((64,))

    @property
    def base_action_type(self) -> RLTypes:
        return RLTypes.ANY

    @property
    def base_observation_type(self) -> RLTypes:
        return RLTypes.CONTINUOUS

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
        return "PPO"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.memory_warmup_size <= self.memory.capacity
        assert self.batch_size < self.memory_warmup_size

    @property
    def info_types(self) -> dict:
        return {
            "policy_loss": {},
            "value_loss": {},
            "entropy_loss": {},
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
class _ActorCriticNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        if config.action_type == RLTypes.CONTINUOUS and config.enable_stable_gradients:
            if config.enable_action_normalization:
                self.mean_low = -1
                self.mean_high = 1
                self.action_range = 2
            else:
                # mean の範囲はactionの取りうる範囲
                self.mean_low = config.action_low
                self.mean_high = config.action_high
                self.action_range = config.action_high - config.action_low

        # Orthogonal initialization and layer scaling
        kernel_initializer = "orthogonal"

        # input
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)

        # image
        if self.in_block.use_image_layer:
            self.image_block = config.image_block.create_block_tf(enable_time_distributed_layer=False)
            self.image_flatten = kl.Flatten()

        # --- hidden block
        self.hidden_block = config.hidden_block.create_block_tf()

        # --- value
        self.value_block = config.value_block.create_block_tf()
        self.value_layer = kl.Dense(1, kernel_initializer=kernel_initializer)

        # --- policy
        self.policy_block = config.policy_block.create_block_tf()
        if self.config.action_type == RLTypes.DISCRETE:
            self.out_layer = kl.Dense(config.action_num, activation="softmax", kernel_initializer=kernel_initializer)
        elif self.config.action_type == RLTypes.CONTINUOUS:
            self.pi_mean_layer = kl.Dense(
                config.action_num,
                activation="linear",
                kernel_initializer=kernel_initializer,
                bias_initializer="truncated_normal",
            )
            self.pi_stddev_layer = kl.Dense(
                config.action_num,
                activation="linear",
                kernel_initializer=kernel_initializer,
                bias_initializer="truncated_normal",
            )
        else:
            raise ValueError(self.action_type)

        # build
        self.build((None,) + config.observation_shape)

    def call(self, state, training=False):
        x = self.in_block(state, training=training)
        if self.in_block.use_image_layer:
            x = self.image_block(x, training=training)
            x = self.image_flatten(x)
        x = self.hidden_block(x, training=training)

        v = self.value_block(x)
        v = self.value_layer(v)

        p = self.policy_block(x)
        if self.config.action_type == RLTypes.DISCRETE:
            prob = self.out_layer(p)
            return v, prob
        elif self.config.action_type == RLTypes.CONTINUOUS:
            mean = self.pi_mean_layer(p)
            stddev = self.pi_stddev_layer(p)

            # σ > 0
            stddev = tf.exp(stddev)

            if self.config.enable_stable_gradients:
                mean = tf.clip_by_value(mean, self.mean_low, self.mean_high)
                stddev = tf.clip_by_value(stddev, 0, self.config.stable_gradients_max_stddev)

            return v, mean, stddev

    def policy(self, state):
        if self.config.action_type == RLTypes.DISCRETE:
            v, probs = self.call(state.reshape(1, -1))
            prob = probs.numpy()[0]
            action = np.random.choice(self.config.action_num, 1, p=prob)[0]
            return v.numpy()[0], prob, action
        elif self.config.action_type == RLTypes.CONTINUOUS:
            v, mean, stddev = self(state.reshape((1, -1)))

            # ガウス分布に従った乱数をだす
            action = tf.random.normal(mean.shape, mean=mean, stddev=stddev)

            if self.config.enable_stable_gradients:
                action = tf.clip_by_value(
                    action,
                    mean - self.action_range,
                    mean + self.action_range,
                )

            return (
                v.numpy()[0],
                mean.numpy()[0],
                stddev.numpy()[0],
                action.numpy()[0],
            )

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def summary(self, name: str = "", **kwargs):
        if hasattr(self.in_block, "init_model_graph"):
            self.in_block.init_model_graph()
        if self.in_block.use_image_layer and hasattr(self.image_block, "init_model_graph"):
            self.image_block.init_model_graph()
        if hasattr(self.hidden_block, "init_model_graph"):
            self.hidden_block.init_model_graph()

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

        self.model = _ActorCriticNetwork(self.config)

        # Adaptive KL penalty
        self.adaptive_kl_beta = 0.5

    def call_restore(self, data: Any, **kwargs) -> None:
        self.model.set_weights(data[0])
        self.adaptive_kl_beta = data[1]

    def call_backup(self, **kwargs) -> Any:
        return [
            self.model.get_weights(),
            self.adaptive_kl_beta,
        ]

    def summary(self, **kwargs):
        self.model.summary(**kwargs)


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter
        self.remote_memory: RemoteMemory = self.remote_memory

        self.train_count = 0

        self.optimizer = keras.optimizers.Adam(learning_rate=self.config.optimizer_initial_lr)

    def get_train_count(self):
        return self.train_count

    def train(self):
        if self.remote_memory.length() < self.config.memory_warmup_size:
            return {}
        batchs = self.remote_memory.sample(self.config.batch_size)

        states = np.asarray([e["state"] for e in batchs])
        advantage = np.asarray([e["discounted_reward"] for e in batchs]).reshape((-1, 1))
        old_v = np.asarray([e["v"] for e in batchs])

        # --- 状態の正規化
        if self.config.enable_state_normalized:
            states = (states - np.mean(states, axis=0, keepdims=True)) / (np.std(states, axis=0, keepdims=True) + 1e-8)

        # --- baseline
        if self.config.baseline_type == "" or self.config.baseline_type == "none":
            pass
        elif self.config.baseline_type == "ave":
            advantage -= np.mean(advantage)
        elif self.config.baseline_type == "std":
            advantage = advantage / (np.std(advantage) + 1e-8)
        elif self.config.baseline_type == "normal":
            advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-8)
        elif self.config.baseline_type == "advantage":
            pass
        else:
            raise ValueError("baseline_type fail. ['none', 'ave', 'std', 'normal', 'advantage]")

        if self.config.action_type == RLTypes.DISCRETE:
            actions = np.asarray([e["action"] for e in batchs])
            old_probs = np.asarray([e["prob"] for e in batchs])

            # アクションをonehotベクトルの形に変形
            onehot_actions = tf.one_hot(actions, self.config.action_num).numpy()

            # old_pi
            old_pi = tf.reduce_sum(onehot_actions * old_probs, axis=1, keepdims=True)
            old_logpi = np.log(old_pi)
        else:
            actions = np.asarray([e["action"] for e in batchs])
            old_logpi = np.asarray([e["logpi"] for e in batchs])
            if self.config.surrogate_type == "kl":
                old_mean = np.asarray([e["mean"] for e in batchs])
                old_stddev = np.asarray([e["stddev"] for e in batchs])

        # --- Qモデルの学習
        with tf.GradientTape() as tape:
            if self.config.action_type == RLTypes.DISCRETE:
                v, new_probs = self.parameter.model(states, training=True)

                # π(a|s)とlog(π(a|s))を計算
                new_pi = tf.reduce_sum(onehot_actions * new_probs, axis=1, keepdims=True)
                new_logpi = tf.math.log(tf.clip_by_value(new_pi, 1e-8, 1.0))

            else:
                v, new_mean, new_stddev = self.parameter.model(states, training=True)
                new_logpi = compute_logprob(new_mean, new_stddev, actions)
                new_pi = tf.exp(new_logpi)

            # (new_pi / old_pi) で計算するとnanになりやすい
            ratio = tf.exp(new_logpi - old_logpi)

            # advantage
            if self.config.baseline_type == "advantage":
                advantage = advantage - tf.stop_gradient(v)

            if self.config.surrogate_type == "clip":
                # Clipped Surrogate Objective
                ratio_clipped = tf.clip_by_value(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)

                # loss の計算
                loss_unclipped = ratio * advantage
                loss_clipped = ratio_clipped * advantage

                # 小さいほうを採用
                policy_loss = tf.minimum(loss_unclipped, loss_clipped)

            elif self.config.surrogate_type == "kl":
                if self.config.action_type == RLTypes.DISCRETE:
                    kl = compute_kl_divergence(old_probs, new_probs)
                else:
                    kl = compute_kl_divergence_normal(old_mean, old_stddev, new_mean, new_stddev)
                policy_loss = ratio * advantage - self.parameter.adaptive_kl_beta * kl
            elif self.config.surrogate_type == "":
                policy_loss = ratio * advantage
            else:
                raise ValueError(self.config.surrogate_type)

            # --- Value loss
            if self.config.enable_value_clip:
                # clipする場合
                v_clipped = tf.clip_by_value(v, old_v - self.config.clip_range, old_v + self.config.clip_range)
                value_loss = tf.maximum((v - advantage) ** 2, (v_clipped - advantage) ** 2)
            else:
                # clipしない場合
                value_loss = (v - advantage) ** 2

            # --- 方策エントロピー
            entropy_loss = tf.reduce_sum(new_pi * new_logpi, axis=1, keepdims=True)

            # --- total loss
            policy_loss = -policy_loss
            value_loss = self.config.value_loss_weight * value_loss
            entropy_loss = -self.config.entropy_weight * entropy_loss

            loss = tf.reduce_mean(policy_loss + value_loss + entropy_loss)  # ミニバッチ
            loss += tf.reduce_sum(self.parameter.model.losses)  # 正則化項

        grads = tape.gradient(loss, self.parameter.model.trainable_variables)
        if self.config.global_gradient_clip_norm != 0:
            grads, _ = tf.clip_by_global_norm(grads, self.config.global_gradient_clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.parameter.model.trainable_variables))

        info = {
            "policy_loss": np.mean(policy_loss.numpy()),
            "value_loss": np.mean(value_loss.numpy()),
            "entropy_loss": np.mean(entropy_loss.numpy()),
        }

        # KLペナルティβの調整
        if self.config.surrogate_type == "kl":
            kl_mean = tf.reduce_mean(kl).numpy()
            if kl_mean < self.config.adaptive_kl_target / 1.5:
                self.parameter.adaptive_kl_beta /= 2
            elif kl_mean > self.config.adaptive_kl_target * 1.5:
                self.parameter.adaptive_kl_beta *= 2

            info["kl_mean"] = kl_mean
            info["kl_beta"] = self.parameter.adaptive_kl_beta
            # nanになる場合は adaptive_kl_target が小さすぎる可能性あり

        # 学習率を減少
        if self.train_count > self.config.optimizer_lr_step:
            lr = self.config.optimizer_final_lr
        else:
            lr = self.config.optimizer_initial_lr - (
                self.config.optimizer_initial_lr - self.config.optimizer_final_lr
            ) * (self.train_count / self.config.optimizer_lr_step)
        self.optimizer.lr = lr
        info["lr"] = lr

        self.train_count += 1
        return info


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter
        self.remote_memory: RemoteMemory = self.remote_memory

        if self.config.enable_action_normalization:
            self.action_center = (self.config.action_space.high + self.config.action_space.low) / 2
            self.action_scale = self.config.action_space.high - self.action_center

    def call_on_reset(self, worker: WorkerRun) -> dict:
        self.recent_batch = []
        self.recent_rewards = []
        self.recent_next_states = []
        return {}

    def call_policy(self, worker: WorkerRun) -> Tuple[Any, dict]:
        state = worker.state
        if self.config.state_clip is not None:
            state = np.clip(state, self.config.state_clip[0], self.config.state_clip[1])

        if self.config.action_type == RLTypes.DISCRETE:
            v, prob, action = self.parameter.model.policy(state)
            action = int(action)
            self.batch = {
                "state": state,
                "action": action,
                "prob": prob,
                "v": v,
            }
        else:
            v, mean, stddev, action = self.parameter.model.policy(state)
            logpi = compute_logprob(mean.reshape((-1, 1)), stddev.reshape((-1, 1)), action.reshape((-1, 1)))
            self.batch = {
                "state": state,
                "action": action,
                "v": v,
                "logpi": logpi[0],
            }

            if self.config.enable_action_normalization:
                action = action * self.action_scale + self.action_center

            # safety action
            action = np.clip(action, self.config.action_low, self.config.action_high)
            action = action.tolist()

            if self.config.surrogate_type == "kl":
                self.batch["mean"] = mean
                self.batch["stddev"] = stddev
            if self.rendering:
                self.batch["mean"] = mean
                self.batch["stddev"] = stddev
                self.batch["env_action"] = action

        return action, {}

    def call_on_step(self, worker: WorkerRun) -> dict:
        if not self.training:
            return {}

        reward = worker.reward

        # 報酬のclip
        if self.config.reward_clip is not None:
            if reward < self.config.reward_clip[0]:
                reward = self.config.reward_clip[0]
            elif reward > self.config.reward_clip[1]:
                reward = self.config.reward_clip[1]

        if self.config.experience_collection_method == "GAE":
            next_state = worker.state
            if self.config.state_clip is not None:
                next_state = np.clip(next_state, self.config.state_clip[0], self.config.state_clip[1])
            self.recent_next_states.append(next_state)

        self.recent_rewards.append(reward)
        self.recent_batch.append(self.batch)

        if worker.done:
            if self.config.experience_collection_method == "MC":
                mc_r = 0
                for i in reversed(range(len(self.recent_batch))):
                    r = self.recent_rewards[i]
                    mc_r = r + self.config.discount * mc_r

                    batch = self.recent_batch[i]
                    batch["discounted_reward"] = mc_r
                    self.remote_memory.add(batch)

            elif self.config.experience_collection_method == "GAE":
                if self.config.action_type == RLTypes.DISCRETE:
                    v, _ = self.parameter.model(np.asarray([e["state"] for e in self.recent_batch]))
                    n_v, _ = self.parameter.model(np.asarray(self.recent_next_states))
                else:
                    v, _, _ = self.parameter.model(np.asarray([e["state"] for e in self.recent_batch]))
                    n_v, _, _ = self.parameter.model(np.asarray(self.recent_next_states))
                v = v.numpy().reshape((-1,))
                n_v = n_v.numpy().reshape((-1,))
                gae = 0
                for i in reversed(range(len(self.recent_batch))):
                    batch = self.recent_batch[i]

                    if i == len(self.recent_batch) - 1:
                        delta = self.recent_rewards[i] - v[i]
                    else:
                        delta = self.recent_rewards[i] + self.config.discount * n_v[i] - v[i]
                    gae = delta + self.config.discount * self.config.gae_discount * gae
                    batch["discounted_reward"] = gae
                    self.remote_memory.add(batch)

            else:
                raise ValueError(self.config.experience_collection_method)

        return {}

    def render_terminal(self, worker, **kwargs) -> None:
        v = self.batch["v"]
        print(f"V: {v[0]:.5f}")

        if self.config.action_type == RLTypes.DISCRETE:
            prob = self.batch["prob"]
            maxa = np.argmax(prob)

            def _render_sub(a: int) -> str:
                s = "{:8.3f}%".format(prob[a])
                return s

            render_discrete_action(maxa, worker.env, self.config, _render_sub)
        else:
            action = self.batch["action"]
            env_action = self.batch["env_action"]
            pi = np.exp(self.batch["logpi"])
            mean = self.batch["mean"]
            stddev = self.batch["stddev"]

            print(f"mean      : {mean}")
            print(f"stddev    : {stddev}")
            print(f"action    : {action}")
            print(f"env_action: {env_action}")
            print(f"pi        : {pi}")
