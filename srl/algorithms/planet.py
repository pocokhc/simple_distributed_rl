import random
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Union, cast

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.rl.algorithms.base_dqn import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import Processor
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, RLConfigComponentExperienceReplayBuffer
from srl.rl.models.config.framework_config import RLConfigComponentFramework
from srl.rl.processors.image_processor import ImageProcessor
from srl.rl.schedulers.scheduler import SchedulerConfig
from srl.utils.common import compare_less_version

kl = keras.layers
tfd = tfp.distributions
v216_older = compare_less_version(tf.__version__, "2.16.0")


"""
paper: https://arxiv.org/abs/1811.04551
ref: https://github.com/danijar/dreamer
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

    #: <:ref:`scheduler`> Learning rate
    lr: Union[float, SchedulerConfig] = 0.001
    batch_length: int = 50

    # Model
    deter_size: int = 200
    stoch_size: int = 30
    num_units: int = 400
    dense_act: Any = "elu"
    cnn_act: Any = "relu"
    cnn_depth: int = 32
    free_nats: float = 3.0
    kl_scale: float = 1.0
    enable_overshooting_loss: bool = False
    max_overshooting_size: int = 5

    # GA
    action_algorithm: str = "ga"  # "ga" or "random"
    pred_action_length: int = 5
    num_generation: int = 10
    num_individual: int = 5
    num_simulations: int = 20
    mutation: float = 0.1
    print_ga_debug: bool = True

    # 経験取得方法
    experience_acquisition_method: str = "episode"  # "episode" or "loop"

    # other
    clip_rewards: str = "none"  # "none" or "tanh"
    dummy_state_val: float = 0.0

    def __post_init__(self):
        super().__post_init__()

    def get_processors(self) -> List[Optional[Processor]]:
        return [
            ImageProcessor(
                image_type=SpaceTypes.COLOR,
                resize=(64, 64),
                enable_norm=True,
            )
        ]

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.IMAGE

    def get_framework(self) -> str:
        return "tensorflow"

    def get_name(self) -> str:
        return "PlaNet"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_memory()
        self.assert_params_framework()


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
class RSSM(keras.Model):
    def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu, **kwargs):
        super().__init__(**kwargs)

        self.gru_cell = kl.GRUCell(deter)
        self.obs1 = kl.Dense(hidden, activation=act)
        self.obs_mean = kl.Dense(stoch, activation=None)
        self.obs_std = kl.Dense(stoch, activation=None)
        self.img1 = kl.Dense(hidden, activation=act)
        self.img2 = kl.Dense(hidden, activation=act)
        self.img_mean = kl.Dense(stoch, activation=None)
        self.img_std = kl.Dense(stoch, activation=None)

    def obs_step(self, prev_stoch, prev_deter, prev_action, embed, training=False, _summary: bool = False):
        deter, prior = self.img_step(prev_stoch, prev_deter, prev_action, training=training, _summary=_summary)
        x = tf.concat([deter, embed], -1)
        x = self.obs1(x)
        mean = self.obs_mean(x)
        std = self.obs_std(x)
        std = tf.nn.softplus(std) + 0.1
        if _summary:
            return [mean, std, prior["mean"], prior["std"]]
        stoch = tfd.MultivariateNormalDiag(mean, std).sample()
        post = {"mean": mean, "std": std, "stoch": stoch}
        return post, deter, prior

    def img_step(self, prev_stoch, prev_deter, prev_action, training=False, _summary: bool = False):
        x = tf.concat([prev_stoch, prev_action], -1)
        x = self.img1(x)
        x, deter = self.gru_cell(x, [prev_deter], training=training)
        deter = deter[0]
        x = self.img2(x)
        mean = self.img_mean(x)
        std = self.img_std(x)
        std = tf.nn.softplus(std) + 0.1
        if _summary:
            return deter, {"mean": mean, "std": std}
        stoch = tfd.MultivariateNormalDiag(mean, std).sample()
        prior = {"mean": mean, "std": std, "stoch": stoch}
        return deter, prior

    def get_initial_state(self, batch_size: int = 1):
        if v216_older:
            return self.gru_cell.get_initial_state(None, batch_size, dtype=self.dtype)
        else:
            return self.gru_cell.get_initial_state(batch_size)[0]

    def build_call(self, config):
        in_stoch = np.zeros((1, config.stoch_size), dtype=config.dtype)
        in_deter = self.get_initial_state()
        in_action = np.zeros((1, config.action_space.n), dtype=config.dtype)
        in_embed = np.zeros((1, 32 * config.cnn_depth), dtype=config.dtype)
        self.obs_step(in_stoch, in_deter, in_action, in_embed)


class ConvEncoder(keras.Model):
    def __init__(self, depth: int = 32, act=tf.nn.relu, **kwargs):
        super().__init__(**kwargs)

        kwargs = dict(kernel_size=4, strides=2, activation=act)
        self.conv1 = kl.Conv2D(filters=1 * depth, **kwargs)
        self.conv2 = kl.Conv2D(filters=2 * depth, **kwargs)
        self.conv3 = kl.Conv2D(filters=4 * depth, **kwargs)
        self.conv4 = kl.Conv2D(filters=8 * depth, **kwargs)
        self.hout = kl.Flatten()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.hout(x)
        return x


class ConvDecoder(keras.Model):
    def __init__(self, depth: int = 32, act=tf.nn.relu, **kwargs):
        super().__init__(**kwargs)

        kwargs = dict(strides=2, activation=act)
        self.in_layer = kl.Dense(32 * depth)
        self.reshape = kl.Reshape([1, 1, 32 * depth])
        self.c1 = kl.Conv2DTranspose(4 * depth, kernel_size=5, **kwargs)
        self.c2 = kl.Conv2DTranspose(2 * depth, kernel_size=5, **kwargs)
        self.c3 = kl.Conv2DTranspose(1 * depth, kernel_size=6, **kwargs)
        self.c4_mean = kl.Conv2DTranspose(3, kernel_size=6, strides=2)
        self.c4_std = kl.Conv2DTranspose(3, kernel_size=6, strides=2)

    def call(self, x, _summary: bool = False):
        x = self.in_layer(x)
        x = self.reshape(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x_mean = self.c4_mean(x)
        x_std = self.c4_std(x)
        x_std = tf.nn.softplus(x_std) + 0.1
        if _summary:
            return x_mean
        return tfd.Independent(
            tfd.Normal(x_mean, x_std),
            reinterpreted_batch_ndims=len(x.shape) - 1,  # type:ignore , ignore check "None"
        )


class DenseDecoder(keras.Model):
    def __init__(self, out_shape, layers: int, units: int, dist: str = "normal", act=tf.nn.elu, **kwargs):
        super().__init__(**kwargs)
        self._out_shape = out_shape
        self._dist = dist

        self.h_layers = [kl.Dense(units, activation=act) for i in range(layers)]
        self.hout_mean = kl.Dense(np.prod(self._out_shape))
        self.hout_std = kl.Dense(np.prod(self._out_shape))

    def call(self, x, _summary: bool = False):
        for layer in self.h_layers:
            x = layer(x)
        x_mean = self.hout_mean(x)
        x_std = self.hout_std(x)
        x_std = tf.nn.softplus(x_std) + 0.1
        x_mean = tf.reshape(x_mean, (-1,) + self._out_shape)
        x_std = tf.reshape(x_std, (-1,) + self._out_shape)
        if _summary:
            return x_mean
        if self._dist == "normal":
            return tfd.Independent(tfd.Normal(x_mean, x_std), reinterpreted_batch_ndims=len(self._out_shape))
        if self._dist == "binary":
            return tfd.Independent(tfd.Bernoulli(x), reinterpreted_batch_ndims=len(self._out_shape))
        raise NotImplementedError(self._dist)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.encode = ConvEncoder(self.config.cnn_depth, self.config.cnn_act)
        self.dynamics = RSSM(self.config.stoch_size, self.config.deter_size, self.config.deter_size)
        self.decode = ConvDecoder(self.config.cnn_depth, self.config.cnn_act)
        self.reward = DenseDecoder((1,), 2, self.config.num_units, "normal", self.config.dense_act, name="Reward")

        # build
        self.encode(np.zeros((1,) + self.config.observation_space.shape, self.config.dtype))
        self.dynamics.build_call(self.config)
        self.decode(np.zeros((1, self.config.deter_size + self.config.stoch_size), self.config.dtype))
        self.reward(np.zeros((1, self.config.deter_size + self.config.stoch_size), self.config.dtype))

    def call_restore(self, data: Any, **kwargs) -> None:
        self.encode.set_weights(data[0])
        self.dynamics.set_weights(data[1])
        self.decode.set_weights(data[2])
        self.reward.set_weights(data[3])

    def call_backup(self, **kwargs) -> Any:
        return [
            self.encode.get_weights(),
            self.dynamics.get_weights(),
            self.decode.get_weights(),
            self.reward.get_weights(),
        ]

    def summary(self, **kwargs):
        self.encode.summary(**kwargs)
        if not v216_older:
            self.dynamics.summary(**kwargs)
        self.decode.summary(**kwargs)
        self.reward.summary(**kwargs)


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_sch = SchedulerConfig.create_scheduler(self.config.lr)

        self.opt_encode = keras.optimizers.Adam(self.lr_sch.get_rate())
        self.opt_dynamics = keras.optimizers.Adam(self.lr_sch.get_rate())
        self.opt_decode = keras.optimizers.Adam(self.lr_sch.get_rate())
        self.opt_reward = keras.optimizers.Adam(self.lr_sch.get_rate())

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        batchs = self.memory.sample()

        if self.config.enable_overshooting_loss:
            self._train_latent_overshooting_loss(batchs)
        else:
            self._train(batchs)

        self.train_count += 1

    def _train(self, batchs):
        states = np.asarray([b["states"] for b in batchs], dtype=np.float32)
        actions = [b["actions"] for b in batchs]
        rewards = np.asarray([b["rewards"] for b in batchs], dtype=np.float32)[..., np.newaxis]

        # onehot action
        actions = tf.one_hot(actions, self.config.action_space.n, axis=2)

        # (batch, seq, shape) -> (batch * seq, shape)
        states = tf.reshape(states, (self.config.batch_size * self.config.batch_length,) + states.shape[2:])
        rewards = tf.reshape(rewards, (self.config.batch_size * self.config.batch_length,) + rewards.shape[2:])

        with tf.GradientTape() as tape:
            embed = self.parameter.encode(states, training=True)
            embed_shape = embed.shape  # type:ignore , ignore check "None"

            # (batch * seq, shape) -> (batch, seq, shape)
            # (batch, seq, shape) -> (seq, batch, shape)
            shape = (self.config.batch_size, self.config.batch_length) + embed_shape[1:]
            embed = tf.reshape(embed, shape)
            embed = tf.transpose(embed, [1, 0, 2])
            actions = tf.transpose(actions, [1, 0, 2])

            stochs = []
            deters = []
            stoch = tf.zeros([self.config.batch_size, self.config.stoch_size], dtype=tf.float32)
            deter = self.parameter.dynamics.get_initial_state(self.config.batch_size)
            post_mean = []
            post_std = []
            prior_mean = []
            prior_std = []
            for i in range(self.config.batch_length):
                post, deter, prior = self.parameter.dynamics.obs_step(
                    stoch, deter, actions[i], embed[i], training=True
                )
                stoch = post["stoch"]
                stochs.append(stoch)
                deters.append(deter)
                post_mean.append(post["mean"])
                post_std.append(post["std"])
                prior_mean.append(prior["mean"])
                prior_std.append(prior["std"])
            stochs = tf.stack(stochs, axis=0)
            deters = tf.stack(deters, axis=0)
            post_mean = tf.stack(post_mean, axis=0)
            post_std = tf.stack(post_std, axis=0)
            prior_mean = tf.stack(prior_mean, axis=0)
            prior_std = tf.stack(prior_std, axis=0)

            # (seq, batch, shape) -> (batch, seq, shape)
            stochs = tf.transpose(stochs, [1, 0, 2])
            deters = tf.transpose(deters, [1, 0, 2])
            post_mean = tf.transpose(post_mean, [1, 0, 2])
            post_std = tf.transpose(post_std, [1, 0, 2])
            prior_mean = tf.transpose(prior_mean, [1, 0, 2])
            prior_std = tf.transpose(prior_std, [1, 0, 2])

            feat = tf.concat([stochs, deters], -1)
            feat_shape = feat.shape  # type:ignore , ignore check "None"
            feat = tf.reshape(feat, (self.config.batch_size * self.config.batch_length,) + feat_shape[2:])
            image_pred = self.parameter.decode(feat)
            reward_pred = self.parameter.reward(feat)

            image_loss = tf.reduce_mean(image_pred.log_prob(states))  # type:ignore , ignore check "None"
            reward_loss = tf.reduce_mean(reward_pred.log_prob(rewards))  # type:ignore , ignore check "None"

            prior_dist = tfd.MultivariateNormalDiag(prior_mean, prior_std)
            post_dist = tfd.MultivariateNormalDiag(post_mean, post_std)

            kl_loss = tfd.kl_divergence(post_dist, prior_dist)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss = tf.maximum(kl_loss, self.config.free_nats)
            loss = self.config.kl_scale * kl_loss - image_loss - reward_loss  # type:ignore , ignore check "None"

            # 正則化項
            loss += tf.reduce_sum(self.parameter.encode.losses)
            loss += tf.reduce_sum(self.parameter.dynamics.losses)
            loss += tf.reduce_sum(self.parameter.decode.losses)
            loss += tf.reduce_sum(self.parameter.reward.losses)

        variables = [
            self.parameter.encode.trainable_variables,
            self.parameter.dynamics.trainable_variables,
            self.parameter.decode.trainable_variables,
            self.parameter.reward.trainable_variables,
        ]
        grads = tape.gradient(loss, variables)
        self.opt_encode.apply_gradients(zip(grads[0], variables[0]))
        self.opt_dynamics.apply_gradients(zip(grads[1], variables[1]))
        self.opt_decode.apply_gradients(zip(grads[2], variables[2]))
        self.opt_reward.apply_gradients(zip(grads[3], variables[3]))

        if self.lr_sch.update(self.train_count):
            lr = self.lr_sch.get_rate()
            self.opt_encode.learning_rate = lr
            self.opt_dynamics.learning_rate = lr
            self.opt_decode.learning_rate = lr
            self.opt_reward.learning_rate = lr

        self.info["img_loss"] = -image_loss.numpy() / (64 * 64 * 3)
        self.info["reward_loss"] = -reward_loss.numpy()
        self.info["kl_loss"] = kl_loss.numpy()

    def _train_latent_overshooting_loss(self, batchs):
        states = np.asarray([b["states"] for b in batchs], dtype=np.float32)
        actions = [b["actions"] for b in batchs]
        rewards = np.asarray([b["rewards"] for b in batchs], dtype=np.float32)[..., np.newaxis]

        # onehot action
        actions = tf.one_hot(actions, self.config.action_space.n, axis=2)

        # (batch, seq, shape) -> (batch * seq, shape)
        states = tf.reshape(states, (self.config.batch_size * self.config.batch_length,) + states.shape[2:])
        rewards = tf.reshape(rewards, (self.config.batch_size * self.config.batch_length,) + rewards.shape[2:])

        with tf.GradientTape() as tape:
            embed = self.parameter.encode(states, training=True)
            embed_shape = embed.shape  # type:ignore , ignore check "None"

            # (batch * seq, shape) -> (batch, seq, shape)
            # (batch, seq, shape) -> (seq, batch, shape)
            shape = (self.config.batch_size, self.config.batch_length) + embed_shape[1:]
            embed = tf.reshape(embed, shape)
            embed = tf.transpose(embed, [1, 0, 2])
            actions = tf.transpose(actions, [1, 0, 2])

            stochs = []
            deters = []
            stoch = tf.zeros([self.config.batch_size, self.config.stoch_size], dtype=tf.float32)
            deter = self.parameter.dynamics.get_initial_state(self.config.batch_size)

            kl_loss_list = []
            overshooting_list = []

            for i in range(self.config.batch_length):
                post, n_deter, prior = self.parameter.dynamics.obs_step(
                    stoch, deter, actions[i], embed[i], training=True
                )

                # image/reward
                stochs.append(post["stoch"])
                deters.append(n_deter)

                # 0step KL loss
                prior_dist = tfd.MultivariateNormalDiag(prior["mean"], prior["std"])
                post_dist = tfd.MultivariateNormalDiag(post["mean"], post["std"])
                step_kl_loss = tfd.kl_divergence(post_dist, prior_dist)

                # calc overshooting KL loss
                n_overshooting_list = [prior]
                for o_prior in overshooting_list:
                    _, o_prior = self.parameter.dynamics.img_step(o_prior["stoch"], deter, actions[i], training=True)
                    o_prior_dist = tfd.MultivariateNormalDiag(o_prior["mean"], o_prior["std"])
                    step_kl_loss += tfd.kl_divergence(post_dist, o_prior_dist)
                    if len(n_overshooting_list) < self.config.max_overshooting_size:
                        n_overshooting_list.append(o_prior)

                # add overshooting KL loss
                step_kl_loss /= len(overshooting_list) + 1
                kl_loss_list.append(step_kl_loss)

                # next
                deter = n_deter
                stoch = post["stoch"]
                overshooting_list = n_overshooting_list

            stochs = tf.stack(stochs, axis=0)
            deters = tf.stack(deters, axis=0)

            # (seq, batch, shape) -> (batch, seq, shape)
            stochs = tf.transpose(stochs, [1, 0, 2])
            deters = tf.transpose(deters, [1, 0, 2])

            feat = tf.concat([stochs, deters], -1)
            feat_shape = feat.shape  # type:ignore , ignore check "None"
            feat = tf.reshape(feat, (self.config.batch_size * self.config.batch_length,) + feat_shape[2:])
            image_pred = self.parameter.decode(feat)
            reward_pred = self.parameter.reward(feat)

            image_loss = tf.reduce_mean(image_pred.log_prob(states))  # type:ignore , ignore check "None"
            reward_loss = tf.reduce_mean(reward_pred.log_prob(rewards))  # type:ignore , ignore check "None"

            kl_loss = tf.reduce_mean(kl_loss_list)
            kl_loss = tf.maximum(kl_loss, self.config.free_nats)
            loss = self.config.kl_scale * kl_loss - image_loss - reward_loss  # type:ignore , ignore check "None"

        variables = [
            self.parameter.encode.trainable_variables,
            self.parameter.dynamics.trainable_variables,
            self.parameter.decode.trainable_variables,
            self.parameter.reward.trainable_variables,
        ]
        grads = tape.gradient(loss, variables)
        self.opt_encode.apply_gradients(zip(grads[0], variables[0]))
        self.opt_dynamics.apply_gradients(zip(grads[1], variables[1]))
        self.opt_decode.apply_gradients(zip(grads[2], variables[2]))
        self.opt_reward.apply_gradients(zip(grads[3], variables[3]))

        if self.lr_sch.update(self.train_count):
            lr = self.lr_sch.get_rate()
            self.opt_encode.learning_rate = lr
            self.opt_dynamics.learning_rate = lr
            self.opt_decode.learning_rate = lr
            self.opt_reward.learning_rate = lr

        self.info["img_loss"] = -image_loss.numpy() / (64 * 64 * 3)
        self.info["reward_loss"] = -reward_loss.numpy()
        self.info["kl_loss"] = kl_loss.numpy()


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.dummy_state = np.full(self.config.observation_space.shape, self.config.dummy_state_val, dtype=np.float32)
        self.screen = None

        self._recent_states = []
        self._recent_actions = []
        self._recent_rewards = []

    def on_reset(self, worker):
        if self.config.experience_acquisition_method != "loop":
            self._recent_states = []
            self._recent_actions = []
            self._recent_rewards = []

        self.deter = self.parameter.dynamics.get_initial_state()
        self.stoch = tf.zeros((1, self.config.stoch_size), dtype=tf.float32)
        self.action = 0

    def policy(self, worker) -> int:
        if self.training:
            self.action = cast(int, self.sample_action())
            return self.action

        # --- rssm step
        embed = self.parameter.encode(worker.state[np.newaxis, ...])
        prev_action = tf.one_hot([self.action], self.config.action_space.n, axis=1)
        latent, deter, _ = self.parameter.dynamics.obs_step(self.stoch, self.deter, prev_action, embed)
        self.feat = tf.concat([latent["stoch"], deter], axis=1)
        self.deter = deter
        self.stoch = latent["stoch"]

        if self.config.action_algorithm == "random":
            self.action = cast(int, self.sample_action())
        elif self.config.action_algorithm == "ga":
            self.action = self._ga_policy(self.deter, self.stoch)
        else:
            raise NotImplementedError(self.config.action_algorithm)

        return int(self.action)

    def _ga_policy(self, deter, stoch):
        # --- 初期個体
        elite_actions = [
            [random.randint(0, self.config.action_space.n - 1) for a in range(self.config.pred_action_length)]
            for _ in range(self.config.num_individual)
        ]
        best_actions = []

        # --- 世代ループ
        for g in range(self.config.num_generation):
            # --- 個体を評価
            t0 = time.time()
            elite_rewards = []
            for i in range(len(elite_actions)):
                rewards = []
                for _ in range(self.config.num_simulations):
                    reward = self._eval_actions(deter, stoch, elite_actions[i])
                    rewards.append(reward)
                elite_rewards.append(np.mean(rewards))
            elite_rewards = np.array(elite_rewards)

            # --- エリート戦略
            next_elite_actions = []
            best_idx = np.random.choice(np.where(elite_rewards == elite_rewards.max())[0])
            best_actions = elite_actions[best_idx]
            next_elite_actions.append(best_actions)

            # debug
            if self.config.print_ga_debug:
                print(f"--- {g}/{self.config.num_generation} {time.time()-t0:.1f}s")
                print(f"*{best_idx} {elite_rewards[best_idx]:.5f} {elite_actions[best_idx]}")
                for idx in range(len(elite_actions)):
                    if idx >= 4:
                        print("...")
                        break
                    print(f" {idx} {elite_rewards[idx]:.3f} {elite_actions[idx]}")

            # 最後は交叉しない
            if self.config.num_generation - 1 == g:
                break

            # weight
            weights = elite_rewards - elite_rewards.min()
            if weights.sum() == 0:
                weights = np.full(len(elite_rewards), 1 / len(elite_rewards))
            else:
                weights = weights / weights.sum()

            # --- 子の作成
            while len(next_elite_actions) < self.config.num_individual:
                # --- 親個体の選択(ルーレット方式、重複あり)
                idx1 = np.argmax(np.random.multinomial(1, weights))
                idx2 = np.argmax(np.random.multinomial(1, weights))

                # --- 一様交叉
                c1 = []
                c2 = []
                for i in range(self.config.pred_action_length):
                    if random.random() < 0.5:
                        _c1 = elite_actions[idx1][i]
                        _c2 = elite_actions[idx2][i]
                    else:
                        _c1 = elite_actions[idx2][i]
                        _c2 = elite_actions[idx1][i]

                    # 突然変異
                    if random.random() < self.config.mutation:
                        _c1 = random.randint(0, self.config.action_space.n - 1)
                    if random.random() < self.config.mutation:
                        _c2 = random.randint(0, self.config.action_space.n - 1)

                    c1.append(_c1)
                    c2.append(_c2)

                next_elite_actions.append(c1)
                next_elite_actions.append(c2)
            elite_actions = next_elite_actions

        # 一番いい結果のアクションを実行
        return best_actions[0]

    def _eval_actions(self, deter, stoch, action_list):
        reward = 0
        for step in range(len(action_list)):
            action = tf.one_hot([action_list[step]], self.config.action_space.n, axis=1)
            deter, prior = self.parameter.dynamics.img_step(stoch, deter, action)
            stoch = prior["stoch"]
            feat = tf.concat([stoch, deter], -1)
            _r = self.parameter.reward(feat).mode()  # type:ignore , ignore check "None"
            reward += _r.numpy()[0][0]
        return reward

    def on_step(self, worker):
        if not self.training:
            return
        next_state = worker.state

        clip_rewards_fn = dict(none=lambda x: x, tanh=tf.tanh)[self.config.clip_rewards]
        reward = clip_rewards_fn(worker.reward)

        if self.config.experience_acquisition_method == "loop":
            self._recent_states.append(next_state)
            self._recent_actions.append(self.action)
            self._recent_rewards.append(reward)
            if len(self._recent_states) == self.config.batch_length:
                self.memory.add(
                    {
                        "states": self._recent_states,
                        "actions": self._recent_actions,
                        "rewards": self._recent_rewards,
                    }
                )
                self._recent_states = []
                self._recent_actions = []
                self._recent_rewards = []
        else:
            if len(self._recent_states) < self.config.batch_length:
                self._recent_states.append(next_state)
                self._recent_actions.append(self.action)
                self._recent_rewards.append(reward)

            if worker.done:
                for _ in range(self.config.batch_length - len(self._recent_states)):
                    self._recent_states.append(next_state)
                    self._recent_actions.append(random.randint(0, self.config.action_space.n - 1))
                    self._recent_rewards.append(reward)

                self.memory.add(
                    {
                        "states": self._recent_states,
                        "actions": self._recent_actions,
                        "rewards": self._recent_rewards,
                    }
                )

    def render_terminal(self, worker, **kwargs) -> None:
        pass

    def render_rgb_array(self, worker, **kwargs) -> Optional[np.ndarray]:
        if self.config.observation_space.stype != SpaceTypes.COLOR:
            return None
        from srl.utils import pygame_wrapper as pw

        state = worker.prev_state

        _view_action = 4
        _view_sample = 3
        IMG_W = 64
        IMG_H = 64
        STR_H = 15
        PADDING = 4
        WIDTH = (IMG_W + PADDING) * _view_action + 5
        HEIGHT = (IMG_H + PADDING + STR_H) * (_view_sample + 1) + STR_H * 2 + 5

        if self.screen is None:
            self.screen = pw.create_surface(WIDTH, HEIGHT)
        pw.draw_fill(self.screen, color=(0, 0, 0))

        # --- decode
        pred_state = self.parameter.decode(self.feat).mode()[0].numpy()  # type:ignore , ignore check "None"
        rmse = np.sqrt(np.mean((state - pred_state) ** 2))

        pred_reward = self.parameter.reward(self.feat).mode()[0][0].numpy()  # type:ignore , ignore check "None"

        img1 = state * 255
        img2 = pred_state * 255

        pw.draw_text(self.screen, 0, 0, "original", color=(255, 255, 255))
        pw.draw_image_rgb_array(self.screen, 0, STR_H, img1)
        pw.draw_text(self.screen, IMG_W + PADDING, 0, f"decode(RMSE: {rmse:.5f})", color=(255, 255, 255))
        pw.draw_image_rgb_array(self.screen, IMG_W + PADDING, STR_H, img2)

        pw.draw_text(self.screen, IMG_W * 2 + PADDING + 10, 10, f"reward: {pred_reward:.4f})", color=(255, 255, 255))

        # 横にアクション後の結果を表示
        invalid_actions = worker.get_invalid_actions()
        i = -1
        for a in range(self.config.action_space.n):
            if a in invalid_actions:
                continue
            i += 1
            if i > _view_action:
                break

            a_str = worker.env.action_to_str(a)
            pw.draw_text(self.screen, (IMG_W + PADDING) * i, 20 + IMG_H, f"action {a_str}", color=(255, 255, 255))

            action = tf.one_hot([a], self.config.action_space.n, axis=1)
            deter, prior = self.parameter.dynamics.img_step(self.stoch, self.deter, action)
            feat = tf.concat([prior["stoch"], deter], axis=1)

            next_state_dist = self.parameter.decode(feat)
            reward_dist = self.parameter.reward(feat)

            # 縦にいくつかサンプルを表示
            for j in range(_view_sample):
                if j == 0:
                    next_state = next_state_dist.mode()  # type:ignore , ignore check "None"
                    reward = reward_dist.mode()  # type:ignore , ignore check "None"
                else:
                    next_state = next_state_dist.sample()  # type:ignore , ignore check "None"
                    reward = reward_dist.sample()  # type:ignore , ignore check "None"

                n_img = next_state[0].numpy() * 255
                reward = reward.numpy()[0][0]

                x = (IMG_W + PADDING) * i
                y = 20 + IMG_H + STR_H + (IMG_H + PADDING + STR_H) * j
                pw.draw_text(self.screen, x, y, f"{reward:.3f}", color=(255, 255, 255))
                pw.draw_image_rgb_array(self.screen, x, y + STR_H, n_img)

        return pw.get_rgb_array(self.screen)
