from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.exception import UndefinedError
from srl.base.rl.config import RLConfig
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import ObservationProcessor
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker import RLWorker
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.rl.functions import helper
from srl.rl.functions.common_tf import compute_kl_divergence, compute_kl_divergence_normal
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, RLConfigComponentExperienceReplayBuffer
from srl.rl.models.config.framework_config import RLConfigComponentFramework
from srl.rl.models.config.mlp_block import MLPBlockConfig
from srl.rl.models.tf.blocks.input_block import create_in_block_out_value
from srl.rl.models.tf.distributions.categorical_dist_block import CategoricalDistBlock
from srl.rl.models.tf.distributions.normal_dist_block import NormalDistBlock
from srl.rl.schedulers.scheduler import SchedulerConfig

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
class Config(
    RLConfig,
    RLConfigComponentExperienceReplayBuffer,
    RLConfigComponentFramework,
):
    """
    <:ref:`RLConfigComponentExperienceReplayBuffer`>
    <:ref:`RLConfigComponentFramework`>
    """

    #: <:ref:`MLPBlock`> hidden layers
    hidden_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())
    #: <:ref:`MLPBlock`> value layers
    value_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())
    #: <:ref:`MLPBlock`> policy layers
    policy_block: MLPBlockConfig = field(init=False, default_factory=lambda: MLPBlockConfig())

    #: 割引報酬の計算方法
    #:
    #: Parameters:
    #:   "MC" : モンテカルロ法
    #:   "GAE": Generalized Advantage Estimator
    experience_collection_method: str = "MC"
    #: discount
    discount: float = 0.9
    #: GAEの割引率
    gae_discount: float = 0.9

    #: baseline
    #:
    #: Parameters:
    #:  "" "none"       : none
    #:  "ave"           : (adv - mean)
    #:  "std"           : adv/std
    #:  "normal"        : (adv - mean)/std
    #:  "advantage" "v" : adv - v
    baseline_type: str = "ave"
    #: surrogate type
    #:
    #: Parameters:
    #:  ""     : none
    #:  "clip" : Clipped Surrogate Objective
    #:  "kl"   : Adaptive KLペナルティ
    surrogate_type: str = "clip"
    #: Clipped Surrogate Objective
    policy_clip_range: float = 0.2
    #: Adaptive KLペナルティ内の定数
    adaptive_kl_target: float = 0.01

    #: value clip flag
    enable_value_clip: float = False
    #: value clip range
    value_clip_range: float = 0.2

    #: <:ref:`scheduler`> Learning rate
    lr: Union[float, SchedulerConfig] = 0.01
    #: 状態価値の反映率
    value_loss_weight: float = 1.0
    #: エントロピーの反映率
    entropy_weight: float = 0.1

    #: 状態の正規化 flag
    enable_state_normalized: bool = False
    #: 勾配のL2におけるclip値(0で無効)
    global_gradient_clip_norm: float = 0.5
    #: 状態のclip(Noneで無効、(-10,10)で指定)
    state_clip: Optional[Tuple[float, float]] = None
    #: 報酬のclip(Noneで無効、(-10,10)で指定)
    reward_clip: Optional[Tuple[float, float]] = None

    #: 勾配爆発の対策, 平均、分散、ランダムアクションで大きい値を出さないようにclipする
    enable_stable_gradients: bool = True
    #: enable_stable_gradients状態での標準偏差のclip
    stable_gradients_stddev_range: tuple = (1e-10, 10)

    def __post_init__(self):
        super().__post_init__()

        self.lr = self.create_scheduler().set_linear(2000, 0.02, 0.01)
        self.memory.capacity = 2000
        self.hidden_block.set((64, 64))
        self.value_block.set((64,))
        self.policy_block.set((64,))

    def get_base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.CONTINUOUS

    def get_framework(self) -> str:
        return "tensorflow"

    def get_processors(self) -> List[Optional[ObservationProcessor]]:
        return [self.input_image_block.get_processor()]

    def get_name(self) -> str:
        return "PPO"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_memory()
        self.assert_params_framework()

    def get_info_types(self) -> dict:
        return {
            "policy_loss": {},
            "value_loss": {},
            "entropy_loss": {},
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
class _ActorCriticNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Orthogonal initialization and layer scaling
        kernel_initializer = "orthogonal"

        # --- input
        self.input_block = create_in_block_out_value(
            config.input_value_block,
            config.input_image_block,
            config.observation_space,
        )

        # --- hidden block
        self.hidden_block = config.hidden_block.create_block_tf()

        # --- value
        self.value_block = config.value_block.create_block_tf()
        self.value_out_layer = kl.Dense(1, kernel_initializer=kernel_initializer)

        # --- policy
        self.policy_block = config.policy_block.create_block_tf()
        if self.config.action_space.stype == SpaceTypes.DISCRETE:
            self.policy_dist_block = CategoricalDistBlock(config.action_space.n)
        elif self.config.action_space.stype == SpaceTypes.CONTINUOUS:
            self.policy_dist_block = NormalDistBlock(
                config.action_space.size,
                enable_stable_gradients=self.config.enable_stable_gradients,
                stable_gradients_stddev_range=self.config.stable_gradients_stddev_range,
            )
        else:
            raise UndefinedError(self.config.action_space)

        # build
        self._in_shape = config.observation_space.shape
        self.build((None,) + self._in_shape)

    def call(self, x, training=False):
        x = self.input_block(x, training=training)
        x = self.hidden_block(x, training=training)

        # value
        v = self.value_block(x, training=training)
        v = self.value_out_layer(v, training=training)

        # policy
        p = self.policy_block(x, training=training)
        p = self.policy_dist_block(p, training=training)
        return v, p

    def policy(self, state):
        v, p = self(state, training=False)
        return v, self.policy_dist_block.get_dist(p)

    @tf.function
    def compute_train_loss(
        self,
        state,
        action,
        advantage,
        old_logpi,
        old_probs,
        old_mean,
        old_stddev,
        old_v,
        adaptive_kl_beta,
    ):
        # --- 現在の方策
        v, p = self(state, training=True)
        policy_dist = self.policy_dist_block.get_grad_dist(p)
        new_logpi = policy_dist.log_prob(action)

        # advantage
        if self.config.baseline_type == "advantage":
            advantage = advantage - tf.stop_gradient(v)

        # --- policy
        kl = 0
        ratio = tf.exp(new_logpi - old_logpi)
        if self.config.surrogate_type == "clip":
            # Clipped Surrogate Objective
            ratio_clipped = tf.clip_by_value(
                ratio, 1 - self.config.policy_clip_range, 1 + self.config.policy_clip_range
            )

            # loss の計算
            loss_unclipped = ratio * advantage
            loss_clipped = ratio_clipped * advantage

            # 小さいほうを採用
            policy_loss = tf.minimum(loss_unclipped, loss_clipped)

        elif self.config.surrogate_type == "kl":
            if self.config.action_space.stype == SpaceTypes.DISCRETE:
                new_probs = policy_dist.probs()
                kl = compute_kl_divergence(old_probs, new_probs)
            elif self.config.action_space.stype == SpaceTypes.CONTINUOUS:
                new_mean = policy_dist.mean()
                new_stddev = policy_dist.stddev()
                kl = compute_kl_divergence_normal(old_mean, old_stddev, new_mean, new_stddev)
            else:
                raise UndefinedError(self.config.action_space.stype)
            policy_loss = ratio * advantage - adaptive_kl_beta * kl
        elif self.config.surrogate_type == "":
            policy_loss = ratio * advantage
        else:
            raise UndefinedError(self.config.surrogate_type)

        # 最大化
        policy_loss = -tf.reduce_mean(policy_loss)

        # --- value loss
        if self.config.enable_value_clip:
            v_clipped = tf.clip_by_value(v, old_v - self.config.value_clip_range, old_v + self.config.value_clip_range)
            value_loss = tf.maximum((v - advantage) ** 2, (v_clipped - advantage) ** 2)
        else:
            value_loss = (v - advantage) ** 2
        # MSEの最小化
        value_loss = self.config.value_loss_weight * tf.reduce_mean(value_loss)

        # --- 方策エントロピーボーナス(最大化)
        # H = Σ-π(a|s)lnπ(a|s)
        entropy_loss = tf.reduce_sum(-tf.exp(new_logpi) * new_logpi, axis=-1)
        entropy_loss = self.config.entropy_weight * -tf.reduce_mean(entropy_loss)

        return policy_loss, value_loss, entropy_loss, kl


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
class Trainer(RLTrainer[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_sch = SchedulerConfig.create_scheduler(self.config.lr)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch.get_rate())

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        batchs = self.memory.sample(self.batch_size)
        self.info = {}

        states = np.asarray([e["state"] for e in batchs])
        advantage = np.asarray([e["discounted_reward"] for e in batchs])[..., np.newaxis]

        # --- 状態の正規化
        if self.config.enable_state_normalized:
            states = (states - np.mean(states, axis=0, keepdims=True)) / (np.std(states, axis=0, keepdims=True) + 1e-8)

        # --- baseline
        if self.config.baseline_type in ["", "none"]:
            pass
        elif self.config.baseline_type == "ave":
            advantage -= np.mean(advantage)
        elif self.config.baseline_type == "std":
            advantage = advantage / (np.std(advantage) + 1e-8)
        elif self.config.baseline_type == "normal":
            advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-8)
        elif self.config.baseline_type in ["advantage", "v"]:
            pass
        else:
            raise UnicodeError("baseline_type fail. ['none', 'ave', 'std', 'normal', 'advantage]")

        # --- old
        old_probs = 0
        old_mean = 0
        old_stddev = 0
        old_v = 0
        if self.config.action_space.stype == SpaceTypes.DISCRETE:
            actions = np.asarray([e["action"] for e in batchs])
            old_logpi = np.asarray([e["log_prob"] for e in batchs])[..., np.newaxis]
            if self.config.surrogate_type == "kl":
                old_probs = np.asarray([e["probs"] for e in batchs])
        else:
            actions = np.asarray([e["action"] for e in batchs])[..., np.newaxis]
            old_logpi = np.asarray([e["log_prob"] for e in batchs])[..., np.newaxis]
            if self.config.surrogate_type == "kl":
                old_mean = np.asarray([e["mean"] for e in batchs])
                old_stddev = np.asarray([e["stddev"] for e in batchs])
        if self.config.enable_value_clip:
            old_v = np.asarray([e["v"] for e in batchs])[..., np.newaxis]

        # --- Qモデルの学習
        with tf.GradientTape() as tape:
            policy_loss, value_loss, entropy_loss, kl = self.parameter.model.compute_train_loss(
                states,
                actions,
                advantage,
                old_logpi,
                old_probs,
                old_mean,
                old_stddev,
                old_v,
                self.parameter.adaptive_kl_beta,
            )
            loss = policy_loss + value_loss + entropy_loss
            loss += tf.reduce_sum(self.parameter.model.losses)  # 正則化項

        grads = tape.gradient(loss, self.parameter.model.trainable_variables)
        if self.config.global_gradient_clip_norm != 0:
            grads, _ = tf.clip_by_global_norm(grads, self.config.global_gradient_clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.parameter.model.trainable_variables))
        self.info["policy_loss"] = policy_loss.numpy()
        self.info["value_loss"] = value_loss.numpy()
        self.info["entropy_loss"] = entropy_loss.numpy()

        # --- KLペナルティβの調整
        if self.config.surrogate_type == "kl":
            kl_mean = tf.reduce_mean(kl).numpy()
            if kl_mean < self.config.adaptive_kl_target / 1.5:
                self.parameter.adaptive_kl_beta /= 2
            elif kl_mean > self.config.adaptive_kl_target * 1.5 and self.parameter.adaptive_kl_beta < 10:
                self.parameter.adaptive_kl_beta *= 2
            self.info["kl_mean"] = kl_mean
            self.info["kl_beta"] = self.parameter.adaptive_kl_beta
            # nanになる場合は adaptive_kl_target が小さすぎる可能性あり

        # lr_schedule
        if self.lr_sch.update(self.train_count):
            lr = self.lr_sch.get_rate()
            self.optimizer.learning_rate = lr
            self.info["lr"] = lr

        self.train_count += 1


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

    def on_reset(self, worker) -> dict:
        self.recent_batch = []
        self.recent_rewards = []
        self.recent_next_states = []
        return {}

    def policy(self, worker) -> Tuple[Any, dict]:
        state = worker.state
        if self.config.state_clip is not None:
            state = np.clip(state, self.config.state_clip[0], self.config.state_clip[1])

        v, policy_dist = self.parameter.model.policy(state[np.newaxis, ...])
        if self.config.action_space.stype == SpaceTypes.DISCRETE:  # int
            act_space = cast(DiscreteSpace, self.config.action_space)
            onehot_action = policy_dist.sample(onehot=True)
            env_action = int(np.argmax(onehot_action))
            batch = {
                "state": state,
                "action": onehot_action.numpy()[0],
                "v": v.numpy()[0][0],
                "log_prob": policy_dist.log_prob(onehot_action).numpy()[0][0],
            }
            if self.config.surrogate_type == "kl" or self.rendering:
                batch["probs"] = policy_dist.probs().numpy()[0]
            self.recent_batch.append(batch)
        elif self.config.action_space.stype == SpaceTypes.CONTINUOUS:  # float,list[float]
            act_space = cast(ArrayContinuousSpace, self.config.action_space)
            action = policy_dist.sample().numpy()[0]
            batch = {
                "state": state,
                "action": action[0],
                "v": v.numpy()[0][0],
                "log_prob": policy_dist.log_prob(action).numpy()[0][0],
            }
            if self.config.surrogate_type == "kl" or self.rendering:
                batch["mean"] = policy_dist.mean().numpy()[0]
                batch["stddev"] = policy_dist.stddev().numpy()[0]
            self.recent_batch.append(batch)

            env_action = np.clip(action, act_space.low, act_space.high)
            env_action = env_action.tolist()
        else:
            raise UndefinedError(self.config.action_space)

        return env_action, {}

    def on_step(self, worker) -> dict:
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

        if worker.done:
            if self.config.experience_collection_method == "MC":
                mc_r = 0
                for i in reversed(range(len(self.recent_batch))):
                    r = self.recent_rewards[i]
                    mc_r = r + self.config.discount * mc_r

                    batch = self.recent_batch[i]
                    batch["discounted_reward"] = np.asarray(mc_r, dtype=np.float32)
                    self.memory.add(batch)

            elif self.config.experience_collection_method == "GAE":
                v, _ = self.parameter.model(np.asarray([e["state"] for e in self.recent_batch]))
                n_v, _ = self.parameter.model(np.asarray(self.recent_next_states))
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
                    batch["discounted_reward"] = np.asarray(gae, dtype=np.float32)
                    self.memory.add(batch)

            else:
                raise UndefinedError(self.config.experience_collection_method)

        return {}

    def render_terminal(self, worker, **kwargs) -> None:
        batch = self.recent_batch[-1]
        v = batch["v"]
        print(f"V: {v:.5f}")

        if self.config.action_space.stype == SpaceTypes.DISCRETE:
            probs = batch["probs"]
            maxa = np.argmax(probs)

            def _render_sub(a: int) -> str:
                s = "{:5.1f}%".format(probs[a] * 100)
                return s

            helper.render_discrete_action(int(maxa), self.config.action_space.n, worker.env, _render_sub)
        elif self.config.action_space.stype == SpaceTypes.CONTINUOUS:
            action = batch["action"]
            prob = np.exp(batch["log_prob"])
            mean = batch["mean"]
            stddev = batch["stddev"]

            print(f"action: {action}")
            print(f"prob  : {prob}")
            print(f"mean  : {mean}")
            print(f"stddev: {stddev}")
        else:
            raise UndefinedError(self.config.action_space.stype)
