import logging
import random
from typing import Any, Optional, cast

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from srl.base.define import DoneTypes, SpaceTypes
from srl.base.exception import UndefinedError
from srl.base.rl.algorithms.base_ppo import RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.trainer import RLTrainer
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.np_array import NpArraySpace
from srl.rl.memories.replay_buffer import RLReplayBuffer
from srl.rl.tf.distributions.bernoulli_dist_block import BernoulliDistBlock
from srl.rl.tf.distributions.categorical_dist_block import CategoricalDistBlock
from srl.rl.tf.distributions.categorical_gumbel_dist_block import CategoricalGumbelDistBlock
from srl.rl.tf.distributions.linear_block import LinearBlock
from srl.rl.tf.distributions.normal_dist_block import NormalDistBlock
from srl.rl.tf.distributions.twohot_dist_block import TwoHotDistBlock
from srl.utils.common import compare_less_version

from .config import Config
from .model import RSSM, ImageDecoder, ImageEncoder, LinearEncoder

v216_older = compare_less_version(tf.__version__, "2.16.0")

logger = logging.getLogger(__name__)


class Memory(RLReplayBuffer):
    def setup(self) -> None:
        super().setup()
        self.seq_action = [[] for _ in range(self.config.batch_size)]
        self.seq_next_state = [[] for _ in range(self.config.batch_size)]
        self.seq_reward = [[] for _ in range(self.config.batch_size)]
        self.seq_undone = [[] for _ in range(self.config.batch_size)]
        self.seq_unterminated = [[] for _ in range(self.config.batch_size)]

        self.register_trainer_recv_func(self.sample_seq)

    def sample_seq(self):
        if self.is_warmup_needed():
            return None

        # --- create sequence batch
        # 各batchにbatch_seq溜まるまでエピソードを追加する
        actions = []
        next_states = []
        rewards = []
        undone = []
        unterminated = []

        def _f(arr1, arr2, i):
            arr1.append(arr2[i][: self.config.batch_length])
            arr2[i] = arr2[i][self.config.batch_length :]

        for i in range(self.config.batch_size):
            while len(self.seq_action[i]) < self.config.batch_length:
                batch = self.sample(batch_size=1)
                assert batch is not None
                batch = cast(dict, batch[0])
                episode_len = len(batch["actions"])
                self.seq_action[i].extend(batch["actions"])
                self.seq_next_state[i].extend(batch["next_states"])
                self.seq_reward[i].extend(batch["rewards"])
                self.seq_undone[i].extend([1 for _ in range(episode_len - 1)] + [0])
                self.seq_unterminated[i].extend([1 for _ in range(episode_len - 1)] + [0 if batch["terminated"] else 1])
            _f(actions, self.seq_action, i)
            _f(next_states, self.seq_next_state, i)
            _f(rewards, self.seq_reward, i)
            _f(undone, self.seq_undone, i)
            _f(unterminated, self.seq_unterminated, i)
        actions = np.asarray(actions, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)[..., np.newaxis]
        undone = np.asarray(undone, dtype=np.float32)[..., np.newaxis]
        unterminated = np.asarray(unterminated, dtype=np.float32)[..., np.newaxis]

        return actions, next_states, rewards, undone, unterminated


class Parameter(RLParameter):
    def setup(self):
        # --- encode/decode
        if self.config.observation_space.is_image_like():
            self.encode = ImageEncoder(
                self.config.observation_space.shape,
                self.config.cnn_depth,
                self.config.cnn_blocks,
                self.config.cnn_activation,
                self.config.cnn_normalization_type,
                self.config.cnn_resize_type,
                self.config.cnn_resized_image_size,
                name="ImageEncoder",
            )
            self.decode = ImageDecoder(
                self.encode,
                self.config.cnn_use_sigmoid,
                self.config.cnn_depth,
                self.config.cnn_blocks,
                self.config.cnn_activation,
                self.config.cnn_normalization_type,
                self.config.cnn_resize_type,
                self.config.encoder_decoder_dist,
                name="ImageDecoder",
            )
            logger.info(f"Encoder/Decoder: Image({self.config.encoder_decoder_dist})")
        else:
            self.encode = LinearEncoder(
                self.config.encoder_decoder_mlp,
                self.config.dense_act,
                name="LinearEncoder",
            )
            if self.config.encoder_decoder_dist == "linear":
                self.decode = LinearBlock(
                    self.config.observation_space.shape[-1],
                    list(reversed(self.config.encoder_decoder_mlp)),
                    activation=self.config.dense_act,
                    use_symlog=self.config.use_symlog,
                    name="LinearDecoder",
                )
                logger.info("Encoder/Decoder: Linear" + ("(symlog)" if self.config.use_symlog else ""))
            elif self.config.encoder_decoder_dist == "normal":
                self.decode = NormalDistBlock(
                    self.config.observation_space.shape[-1],
                    list(reversed(self.config.encoder_decoder_mlp)),
                    activation=self.config.dense_act,
                    name="LinearDecoder",
                )
                logger.info("Encoder/Decoder: Normal")
            else:
                raise UndefinedError(self.config.encoder_decoder_dist)

        # --- dynamics
        self.dynamics = RSSM(
            self.config.rssm_deter_size,
            self.config.rssm_stoch_size,
            self.config.rssm_classes,
            self.config.rssm_hidden_units,
            self.config.rssm_unimix,
            self.config.rssm_activation,
            self.config.rssm_use_norm_layer,
            self.config.rssm_use_categorical_distribution,
            name="RSSM",
        )

        # --- reward
        if self.config.reward_type == "linear":
            self.reward = LinearBlock(
                1,
                self.config.reward_layer_sizes,
                self.config.dense_act,
                use_symlog=self.config.use_symlog,
                name="RewardModel",
            )
            logger.info("Reward: Linear" + ("(symlog)" if self.config.use_symlog else ""))
        elif self.config.reward_type == "twohot":
            self.reward = TwoHotDistBlock(
                self.config.reward_twohot_bins,
                self.config.reward_twohot_low,
                self.config.reward_twohot_high,
                self.config.reward_layer_sizes,
                self.config.dense_act,
                use_symlog=self.config.use_symlog,
                name="RewardModel",
            )
            logger.info("Reward: TwoHot" + ("(symlog)" if self.config.use_symlog else ""))
        elif self.config.reward_type == "normal":
            self.reward = NormalDistBlock(1, self.config.reward_layer_sizes, (), (), self.config.dense_act, name="RewardModel")
            logger.info("Reward: Normal")
        elif self.config.reward_type == "normal_fixed_scale":
            self.reward = NormalDistBlock(1, self.config.reward_layer_sizes, (), (), self.config.dense_act, 1, name="RewardModel")
            logger.info("Reward: Normal(stddev=1)")
        else:
            raise UndefinedError(self.config.reward_type)

        # --- continue
        self.cont = BernoulliDistBlock(self.config.cont_layer_sizes, self.config.dense_act, name="ContinueModel")
        logger.info("Continue: Bernoulli")

        # --- critic
        logger.info(f"Critic estimation method: {self.config.critic_estimation_method}")
        if self.config.critic_type == "linear":
            self.critic = LinearBlock(
                1,
                self.config.critic_layer_sizes,
                self.config.dense_act,
                self.config.use_symlog,
                name="Critic",
            )
            self.critic_target = LinearBlock(
                1,
                self.config.critic_layer_sizes,
                self.config.dense_act,
                self.config.use_symlog,
                name="CriticTarget",
            )
            logger.info("Critic: Linear" + ("(symlog)" if self.config.use_symlog else ""))
        elif self.config.critic_type == "normal":
            self.critic = NormalDistBlock(1, self.config.critic_layer_sizes, (), (), self.config.dense_act, name="Critic")
            self.critic_target = NormalDistBlock(1, self.config.critic_layer_sizes, (), (), self.config.dense_act, name="CriticTarget")
            logger.info("Critic: Normal")
        elif self.config.critic_type == "normal_fixed_scale":
            self.critic = NormalDistBlock(
                1,
                self.config.critic_layer_sizes,
                activation=self.config.dense_act,
                fixed_scale=1,
                name="Critic",
            )
            self.critic_target = NormalDistBlock(
                1,
                self.config.critic_layer_sizes,
                activation=self.config.dense_act,
                fixed_scale=1,
                name="CriticTarget",
            )
            logger.info("Critic: Normal(stddev=1)")
        elif self.config.critic_type == "twohot":
            self.critic = TwoHotDistBlock(
                self.config.critic_twohot_bins,
                self.config.critic_twohot_low,
                self.config.critic_twohot_high,
                self.config.critic_layer_sizes,
                self.config.dense_act,
                use_symlog=self.config.use_symlog,
                name="Critic",
            )
            self.critic_target = TwoHotDistBlock(
                self.config.critic_twohot_bins,
                self.config.critic_twohot_low,
                self.config.critic_twohot_high,
                self.config.critic_layer_sizes,
                self.config.dense_act,
                use_symlog=self.config.use_symlog,
                name="CriticTarget",
            )
            logger.info("Critic: TwoHot" + ("(symlog)" if self.config.use_symlog else ""))
        else:
            raise UndefinedError(self.config.critic_type)

        # --- actor
        logger.info(f"Actor loss: {self.config.actor_loss_type}")
        if isinstance(self.config.action_space, DiscreteSpace):
            if self.config.actor_discrete_type == "categorical":
                self.actor = CategoricalDistBlock(
                    self.config.action_space.n,
                    self.config.actor_layer_sizes,
                    name="Actor",
                )
                logger.info(f"Actor: Categorical(unimix={self.config.actor_discrete_unimix})")
            elif self.config.actor_discrete_type == "gumbel_categorical":
                self.actor = CategoricalGumbelDistBlock(
                    self.config.action_space.n,
                    self.config.actor_layer_sizes,
                    name="Actor",
                )
                logger.info("Actor: GumbelCategorical")
            else:
                raise UndefinedError(self.config.actor_discrete_type)
        elif isinstance(self.config.action_space, NpArraySpace):
            self.actor = NormalDistBlock(
                self.config.action_space.size,
                self.config.actor_layer_sizes,
                name="Actor",
            )
            logger.info("Actor: Normal")
        else:
            raise ValueError(self.config.action_space)

        # --- build
        self.encode(np.zeros((1,) + self.config.observation_space.shape, self.config.dtype))
        embed: Any = self.dynamics.build_call(self.config, self.encode.out_size)
        self.decode(np.zeros((1, embed.shape[1]), self.config.dtype))
        self.reward(np.zeros((1, embed.shape[1]), self.config.dtype))
        self.cont(np.zeros((1, embed.shape[1]), self.config.dtype))
        self.critic(np.zeros((1, embed.shape[1]), self.config.dtype))
        self.critic_target(np.zeros((1, embed.shape[1]), self.config.dtype))
        self.actor(np.zeros((1, embed.shape[1]), self.config.dtype))

        # --- sync target
        self.critic_target.set_weights(self.critic.get_weights())

    def call_restore(self, data: Any, **kwargs) -> None:
        self.encode.set_weights(data[0])
        self.dynamics.set_weights(data[1])
        self.decode.set_weights(data[2])
        self.reward.set_weights(data[3])
        self.cont.set_weights(data[4])
        self.critic.set_weights(data[5])
        self.critic_target.set_weights(data[5])
        self.actor.set_weights(data[6])

    def call_backup(self, **kwargs) -> Any:
        return [
            self.encode.get_weights(),
            self.dynamics.get_weights(),
            self.decode.get_weights(),
            self.reward.get_weights(),
            self.cont.get_weights(),
            self.critic.get_weights(),
            self.actor.get_weights(),
        ]

    def summary(self, **kwargs):
        self.encode.summary(**kwargs)
        if not v216_older:
            self.dynamics.summary(**kwargs)
        self.decode.summary(**kwargs)
        self.reward.summary(**kwargs)
        self.cont.summary(**kwargs)
        self.critic.summary(**kwargs)
        self.actor.summary(**kwargs)


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.opt_encode = keras.optimizers.Adam(learning_rate=self.config.lr_model_scheduler.apply_tf_scheduler(self.config.lr_model))
        self.opt_decode = keras.optimizers.Adam(learning_rate=self.config.lr_model_scheduler.apply_tf_scheduler(self.config.lr_model))
        self.opt_dynamics = keras.optimizers.Adam(learning_rate=self.config.lr_model_scheduler.apply_tf_scheduler(self.config.lr_model))
        self.opt_reward = keras.optimizers.Adam(learning_rate=self.config.lr_model_scheduler.apply_tf_scheduler(self.config.lr_model))
        self.opt_cont = keras.optimizers.Adam(learning_rate=self.config.lr_model_scheduler.apply_tf_scheduler(self.config.lr_model))
        self.opt_critic = keras.optimizers.Adam(learning_rate=self.config.lr_critic_scheduler.apply_tf_scheduler(self.config.lr_critic))
        self.opt_actor = keras.optimizers.Adam(learning_rate=self.config.lr_actor_scheduler.apply_tf_scheduler(self.config.lr_actor))

        self.stoch, self.deter = self.parameter.dynamics.get_initial_state(self.config.batch_size)

    def train(self) -> None:
        batches = self.memory.sample_seq()
        if batches is None:
            return
        self.train_count += 1

        actions, next_states, rewards, undone, unterminated = batches

        # (batch, seq, shape) -> (seq, batch, shape)
        actions = tf.transpose(actions, [1, 0, 2])
        # (batch, seq, shape) -> (seq, batch, shape) -> (seq*batch, shape)
        _t = list(range(len(next_states.shape)))
        _t[0], _t[1] = _t[1], _t[0]
        next_states = tf.reshape(
            tf.transpose(next_states, _t),
            (self.config.batch_size * self.config.batch_length,) + next_states.shape[2:],
        )
        rewards = tf.reshape(
            tf.transpose(rewards, [1, 0, 2]),
            (self.config.batch_size * self.config.batch_length,) + rewards.shape[2:],
        )
        undone = tf.reshape(
            tf.transpose(undone, [1, 0, 2]),
            (self.config.batch_size * self.config.batch_length,) + undone.shape[2:],
        )
        unterminated = tf.reshape(
            tf.transpose(unterminated, [1, 0, 2]),
            (self.config.batch_size * self.config.batch_length,) + unterminated.shape[2:],
        )

        # ------------------------
        # RSSM
        # ------------------------
        self.parameter.encode.trainable = True
        self.parameter.decode.trainable = True
        self.parameter.dynamics.trainable = True
        self.parameter.reward.trainable = True
        self.parameter.cont.trainable = True
        self.parameter.actor.trainable = False
        self.parameter.critic.trainable = False
        with tf.GradientTape() as tape:
            embed = self.parameter.encode(next_states, training=True)
            (
                stochs,
                deters,
                feats,
                kl_loss_dyn,
                kl_loss_rep,
                self.stoch,
                self.deter,
            ) = self.parameter.dynamics.compute_train_loss(
                embed,
                actions,
                self.stoch,
                self.deter,
                undone,
                self.config.batch_size,
                self.config.batch_length,
                self.config.free_nats,
            )

            # --- embed loss
            decode_loss = self.parameter.decode.compute_train_loss(feats, next_states)
            reward_loss = self.parameter.reward.compute_train_loss(feats, rewards)
            cont_loss = self.parameter.cont.compute_train_loss(feats, unterminated)

            loss = self.config.loss_scale_pred * (decode_loss + reward_loss + cont_loss) + self.config.loss_scale_kl_dyn * kl_loss_dyn + self.config.loss_scale_kl_rep * kl_loss_rep
        self.info["decode_loss"] = np.mean(decode_loss.numpy())
        self.info["reward_loss"] = np.mean(reward_loss.numpy())
        self.info["cont_loss"] = np.mean(cont_loss.numpy())
        self.info["kl_loss"] = np.mean(kl_loss_dyn.numpy())

        if self.config.enable_train_model:
            variables = [
                self.parameter.encode.trainable_variables,
                self.parameter.dynamics.trainable_variables,
                self.parameter.decode.trainable_variables,
                self.parameter.reward.trainable_variables,
                self.parameter.cont.trainable_variables,
            ]
            grads = tape.gradient(loss, variables)
            self.opt_encode.apply_gradients(zip(grads[0], variables[0]))
            self.opt_dynamics.apply_gradients(zip(grads[1], variables[1]))
            self.opt_decode.apply_gradients(zip(grads[2], variables[2]))
            self.opt_reward.apply_gradients(zip(grads[3], variables[3]))
            self.opt_cont.apply_gradients(zip(grads[4], variables[4]))

        if (not self.config.enable_train_actor) and (not self.config.enable_train_critic):
            # WorldModelsのみ学習
            return
        if self.train_count < self.config.warmup_world_model:
            return

        self.parameter.encode.trainable = False
        self.parameter.decode.trainable = False
        self.parameter.dynamics.trainable = False
        self.parameter.reward.trainable = False
        self.parameter.cont.trainable = False

        # ------------------------
        # Actor
        # ------------------------
        horizon_feat = None
        horizon_V = None
        if self.config.enable_train_actor:
            self.parameter.actor.trainable = True
            self.parameter.critic.trainable = False
            with tf.GradientTape() as tape:
                actor_loss, act_v_loss, entropy_loss, horizon_feat, horizon_V = self._compute_horizon_step(stochs, deters, feats)
            grads = tape.gradient(actor_loss, self.parameter.actor.trainable_variables)
            self.opt_actor.apply_gradients(zip(grads, self.parameter.actor.trainable_variables))
            if act_v_loss is not None:
                self.info["act_v_loss"] = act_v_loss.numpy()
            if entropy_loss is not None:
                self.info["entropy_loss"] = entropy_loss.numpy()
            self.info["actor_loss"] = actor_loss.numpy()

        # ------------------------
        # critic
        # ------------------------
        if self.config.enable_train_critic:
            if horizon_feat is None:
                actor_loss, act_v_loss, entropy_loss, horizon_feat, horizon_V = self._compute_horizon_step(stochs, deters, feats, is_critic=True)

            self.parameter.actor.trainable = False
            self.parameter.critic.trainable = True
            with tf.GradientTape() as tape:
                critic_loss = self.parameter.critic.compute_train_loss(horizon_feat, tf.stop_gradient(horizon_V))
            grads = tape.gradient(critic_loss, self.parameter.critic.trainable_variables)
            self.opt_critic.apply_gradients(zip(grads, self.parameter.critic.trainable_variables))
            self.info["critic_loss"] = critic_loss.numpy()

            # --- target update
            if self.config.critic_target_update_interval > 0:
                if self.train_count % self.config.critic_target_update_interval == 0:
                    self.parameter.critic_target.set_weights(self.parameter.critic.get_weights())
                else:
                    self.parameter.critic_target.set_weights((1 - self.config.critic_target_soft_update) * np.array(self.parameter.critic.get_weights(), dtype=object) + (self.config.critic_target_soft_update) * np.array(self.parameter.critic.get_weights(), dtype=object))

        return

    @tf.function
    def _compute_horizon_step(self, stoch, deter, feat, is_critic: bool = False):
        # featはアクション後の状態でQに近いイメージ
        horizon_feat = [feat]
        horizon_logpi = []
        entropy = tf.zeros(stoch.shape[0])
        for t in range(self.config.horizon):
            # --- calc action
            if isinstance(self.config.action_space, DiscreteSpace):
                dist = self.parameter.actor(feat)
                action = dist.rsample()
                if self.config.horizon_policy == "random":
                    action = tf.one_hot(
                        np.random.randint(0, self.config.action_space.n - 1, size=stoch.shape[0]),
                        self.config.action_space.n,
                    )
                log_probs = dist.log_probs()
                entropy += -tf.reduce_sum(tf.exp(log_probs) * log_probs, axis=-1)
            elif isinstance(self.config.action_space, NpArraySpace):
                dist = self.parameter.actor(feat)
                action = dist.rsample()
                if self.config.actor_continuous_enable_normal_squashed:
                    log_prob = dist.log_prob_sgp(action)
                    action = tf.tanh(action)
                else:
                    log_prob = dist.log_prob(action)
                horizon_logpi.append(log_prob)
                entropy += -tf.squeeze(log_prob, axis=-1)
                if self.config.horizon_policy == "random":
                    action = tf.random.normal(action.shape)
            else:
                raise UndefinedError(self.config.action_space)

            # --- rssm step
            deter, prior = self.parameter.dynamics.img_step(stoch, deter, action)
            stoch = prior["stoch"]
            feat = tf.concat([stoch, deter], -1)
            horizon_feat.append(feat)
        horizon_feat = tf.stack(horizon_feat)
        horizon_logpi = tf.stack(horizon_logpi)

        horizon_reward = self.parameter.reward(horizon_feat).mode()
        horizon_cont = tf.cast(self.parameter.cont(horizon_feat).mode(), dtype=tf.float32)
        if self.config.critic_target_update_interval > 0:
            horizon_v = self.parameter.critic_target(horizon_feat).mode()
        else:
            horizon_v = self.parameter.critic(horizon_feat).mode()

        # --- compute V
        # (horizon+1, batch_size*batch_length, shape) -> (batch_size*batch_length, shape)
        horizon_V = _compute_V(
            self.config.critic_estimation_method,
            horizon_reward,
            horizon_v,
            horizon_cont,
            self.config.discount,
            self.config.horizon_ewa_disclam,
            self.config.horizon_h_return,
        )
        if is_critic:
            return None, None, None, horizon_feat, horizon_V

        # --- compute actor
        act_v_loss = None
        entropy_loss = None
        if self.config.actor_loss_type == "dreamer_v1":
            # Vの最大化
            actor_loss = -tf.reduce_mean(tf.reduce_sum(horizon_V[1:], axis=0))
        elif self.config.actor_loss_type == "dreamer_v2":
            adv = horizon_V[1:]

            if self.config.action_space.stype == SpaceTypes.DISCRETE:
                # dynamics backprop 最大化
                act_v_loss = -tf.reduce_mean(tf.reduce_sum(adv, axis=0))
            elif self.config.action_space.stype == SpaceTypes.CONTINUOUS:
                # reinforce 最大化
                if self.config.reinforce_baseline == "v":
                    adv = adv - horizon_v[1:]
                # advの勾配のみ流す、horizon_logpiは勾配が流れていると思うけど学習できなかった
                adv = tf.reduce_sum(
                    adv * self.config.actor_reinforce_rate + (1 - self.config.actor_reinforce_rate) * horizon_logpi * tf.stop_gradient(adv),
                    axis=0,
                )
                act_v_loss = -tf.reduce_mean(adv)

            else:
                raise UndefinedError(self.config.action_space.stype)

            # entropyの最大化
            entropy_loss = -self.config.entropy_rate * tf.reduce_mean(entropy)

            actor_loss = act_v_loss + entropy_loss

        elif self.config.actor_loss_type == "dreamer_v3":
            adv = horizon_V[1:]

            # パーセンタイルの計算
            d5 = tfp.stats.percentile(adv, 5)
            d95 = tfp.stats.percentile(adv, 95)
            adv = adv / tf.maximum(1.0, d95 - d5)

            # dynamics backprop 最大化
            act_v_loss = -tf.reduce_mean(tf.reduce_sum(adv, axis=0))

            # entropyの最大化
            entropy_loss = -self.config.entropy_rate * tf.reduce_mean(entropy)

            actor_loss = act_v_loss + entropy_loss
        else:
            raise UndefinedError(self.config.actor_loss_type)

        return actor_loss, act_v_loss, entropy_loss, horizon_feat, horizon_V


def _compute_V(
    critic_estimation_method: str,
    horizon_reward,
    horizon_v,
    horizon_cont,
    discount: float,
    horizon_ewa_disclam: float,
    horizon_h_return: float,
) -> Any:
    horizon = horizon_reward.shape[0]
    batch = horizon_reward.shape[1]

    cont1 = []
    cont2 = []
    done = []
    cont3 = []
    rewards = []
    _is_cont = tf.ones((batch, 1), dtype=tf.float32)
    for t in range(horizon):
        rewards.append(horizon_reward[t] * _is_cont)
        cont1.append(_is_cont)
        done.append((1 - horizon_cont[t]) * _is_cont)
        if t == horizon - 1:
            cont3.append(_is_cont)
        else:
            cont3.append(tf.zeros((batch, 1), dtype=tf.float32))

        # --- 一度終了したらその後はなし
        _is_cont = _is_cont * horizon_cont[t]
        cont2.append(_is_cont)

    if critic_estimation_method == "simple":
        horizon_V = tf.math.cumsum(rewards, reverse=True)
    elif critic_estimation_method == "discount":
        horizon_V = []
        v = tf.zeros((batch, 1), dtype=tf.float32)
        for t in reversed(range(horizon)):
            v = rewards[t] + cont2[t] * discount * (horizon_v[t] * cont3[t] + v)
            horizon_V.insert(0, v)
        horizon_V = tf.stack(horizon_V)
    elif critic_estimation_method == "ewa":
        VN = []
        v = tf.zeros((batch, 1), dtype=tf.float32)
        for t in reversed(range(horizon)):
            v = rewards[t] + cont2[t] * discount * (horizon_v[t] * cont3[t] + v)
            VN.insert(0, v)

        horizon_V = []
        for t in range(horizon):
            v = VN[t]
            for t2 in range(t + 1, horizon):
                v = (1 - horizon_ewa_disclam) * v + horizon_ewa_disclam * VN[t2]
            horizon_V.append(v)
        horizon_V = tf.stack(horizon_V)

    elif critic_estimation_method == "h-return":
        horizon_V = []
        v = tf.zeros((batch, 1), dtype=tf.float32)
        for t in reversed(range(horizon)):
            a = (1 - horizon_h_return) * horizon_v[t] + horizon_h_return * v
            b = horizon_v[t]
            v = rewards[t] + cont2[t] * discount * (a * (1.0 - cont3[t]) + b * cont3[t])
            horizon_V.insert(0, v)
        horizon_V = tf.stack(horizon_V)
    else:
        raise UndefinedError(critic_estimation_method)

    return horizon_V


class Worker(RLWorker[Config, Parameter, Memory]):
    def on_setup(self, worker, context) -> None:
        self.screen = None

    def on_reset(self, worker):
        self.stoch, self.deter = self.parameter.dynamics.get_initial_state()
        if isinstance(self.config.action_space, DiscreteSpace):
            self.action = tf.one_hot([0], self.config.action_space.n, dtype=tf.float32)
        elif isinstance(self.config.action_space, NpArraySpace):
            self.action = tf.constant([[0 for _ in range(self.config.action_space.size)]], dtype=tf.float32)
        else:
            raise UndefinedError(self.config.action_space.stype)
        # 初期状態へのstep
        state = worker.state.astype(np.float32)
        self._recent_actions = [self.action.numpy()[0]]
        self._recent_next_states = [state]
        self._recent_rewards = [0]
        self._rssm_step(state, self.action)

    def _rssm_step(self, state, action):
        embed = self.parameter.encode(state[np.newaxis, ...])
        deter, prior = self.parameter.dynamics.img_step(self.stoch, self.deter, action)
        post = self.parameter.dynamics.obs_step(deter, embed)
        self.feat = tf.concat([post["stoch"], deter], axis=1)
        self.deter = deter
        self.stoch = post["stoch"]

    def policy(self, worker):
        # debug
        if random.random() < self.config.epsilon:
            env_action = self.sample_action()
            if isinstance(self.config.action_space, DiscreteSpace):
                self.action = tf.one_hot([env_action], self.config.action_space.n, dtype=tf.float32)
            elif isinstance(self.config.action_space, NpArraySpace):
                self.action = tf.constant([env_action], dtype=tf.float32)
            else:
                raise UndefinedError(self.config.action_space.stype)
            return env_action

        dist = self.parameter.actor(self.feat)
        if isinstance(self.config.action_space, DiscreteSpace):
            self.action = dist.sample(onehot=True)
            env_action = int(np.argmax(self.action[0]))
        elif isinstance(self.config.action_space, NpArraySpace):
            act_space = cast(NpArraySpace, self.config.action_space)

            self.action = dist.sample()
            env_action = self.action[0].numpy()
            if self.config.actor_continuous_enable_normal_squashed:
                self.logpi = dist.log_prob_sgp(self.action)
                # Squashed Gaussian Policy (-1, 1) -> (action range)
                env_action = np.tanh(env_action)
                env_action = (1 + env_action) / 2
                env_action = act_space.low + env_action * (act_space.high - act_space.low)
            else:
                self.logpi = dist.log_prob(self.action)
                env_action = env_action * (act_space.high - act_space.low) + act_space.low
                env_action = np.clip(env_action, act_space.low, act_space.high)
        else:
            raise UndefinedError(self.config.action_space)

        return env_action

    def on_step(self, worker):
        next_state = worker.next_state.astype(np.float32)
        self._rssm_step(next_state, self.action)

        if not self.training:
            return

        clip_rewards_fn = dict(none=lambda x: x, tanh=tf.tanh)[self.config.clip_rewards]
        reward = clip_rewards_fn(worker.reward)

        # 1episodeのbatch
        self._recent_actions.append(self.action.numpy()[0])
        self._recent_next_states.append(next_state)
        self._recent_rewards.append(reward)

        if worker.done:
            batch = {
                "actions": self._recent_actions,
                "next_states": self._recent_next_states,
                "rewards": self._recent_rewards,
                "terminated": worker.done_type == DoneTypes.TERMINATED,
            }
            self.memory.add(batch)

    def render_terminal(self, worker, **kwargs) -> None:
        # --- decode
        pred_state = self.parameter.decode(self.feat).sample()[0].numpy()
        pred_reward = self.parameter.reward(self.feat).mode()[0][0].numpy()
        pred_cont = self.parameter.cont(self.feat).prob()[0][0].numpy()
        value = self.parameter.critic(self.feat).mode()[0][0].numpy()

        print(pred_state)
        print(f"reward: {pred_reward:.5f}, done: {(1 - pred_cont) * 100:4.1f}%, v: {value:.5f}")

        if isinstance(self.config.action_space, DiscreteSpace):
            act_dist = self.parameter.actor(self.feat)
            act_probs = act_dist.probs().numpy()[0]
            maxa = np.argmax(act_probs)

            def _render_sub(a: int) -> str:
                # rssm step
                action = tf.one_hot([a], self.config.action_space.n, axis=1)  # type: ignore
                deter, prior = self.parameter.dynamics.img_step(self.stoch, self.deter, action)
                feat = tf.concat([prior["stoch"], deter], axis=1)

                # サンプルを表示
                next_state = self.parameter.decode(feat).sample()
                reward = self.parameter.reward(feat).mode()
                cont = self.parameter.cont(feat).sample()
                # cont = self.parameter.cont(feat).prob()
                value = self.parameter.critic(feat).mode()
                s = f"{act_probs[a] * 100:4.1f}%"
                s += f", {next_state[0].numpy()}"
                s += f", reward {reward.numpy()[0][0]:.5f}"
                s += f", done {(1 - cont.numpy()[0][0]) * 100:4.1f}%"
                s += f", value {value.numpy()[0][0]:.5f}"
                return s

            worker.print_discrete_action_info(int(maxa), _render_sub)
        elif isinstance(self.config.action_space, NpArraySpace):
            act_dist = cast(NormalDistBlock, self.parameter.actor)(self.feat)
            action = act_dist.mean()
            deter, prior = self.parameter.dynamics.img_step(self.stoch, self.deter, action)
            feat = tf.concat([prior["stoch"], deter], axis=1)

            next_state = self.parameter.decode(feat).sample()
            reward = self.parameter.reward(feat).mode()
            cont = self.parameter.cont(feat).prob()
            value = self.parameter.critic(feat).mode()

            print(f"act   : {action.numpy()[0]}, mean {act_dist.mean().numpy()[0]}, std {act_dist.stddev().numpy()[0]}")
            print(f"next  : {next_state[0].numpy()}")
            print(f"reward: {reward[0][0].numpy()}")
            print(f"done  : {(1 - cont[0][0].numpy()) * 100:4.1f}%")
            print(f"value : {value[0].numpy()}")

        else:
            raise ValueError(self.config.action_space.stype)

    def render_rgb_array(self, worker, **kwargs) -> Optional[np.ndarray]:
        if not self.config.observation_space.is_image_like():
            return None
        state = worker.state

        from srl.utils import pygame_wrapper as pw

        _view_action = 4
        _view_sample = 3
        IMG_W = 64
        IMG_H = 64
        STR_H = 20
        PADDING = 8
        WIDTH = (IMG_W + PADDING) * _view_action + 5
        HEIGHT = (IMG_H + PADDING + STR_H * 3) * (_view_sample + 1) + 5

        if self.screen is None:
            self.screen = pw.create_surface(WIDTH, HEIGHT)
        pw.draw_fill(self.screen, color=(0, 0, 0))

        # --- decode
        pred_state = self.parameter.decode(self.feat).sample()[0].numpy()
        pred_reward = self.parameter.reward(self.feat).mode()[0][0].numpy()
        pred_cont = self.parameter.cont(self.feat).prob()[0][0].numpy()
        value = self.parameter.critic(self.feat).mode()[0][0].numpy()
        rmse = np.sqrt(np.mean((state - pred_state) ** 2))

        img1 = np.clip(state * 255, 0, 255).astype(np.int64)
        img2 = np.clip(pred_state * 255, 0, 255).astype(np.int64)

        pw.draw_text(self.screen, 0, 0, "original", color=(255, 255, 255))
        pw.draw_image_rgb_array(self.screen, 0, STR_H, img1)
        pw.draw_text(
            self.screen,
            IMG_W + PADDING,
            0,
            f"decode(RMSE: {rmse:.2f})",
            color=(255, 255, 255),
        )
        pw.draw_image_rgb_array(self.screen, IMG_W + PADDING, STR_H, img2)

        pw.draw_text(
            self.screen,
            IMG_W * 2 + PADDING + 10,
            14,
            f"reward: {pred_reward:.2f}",
            color=(255, 255, 255),
        )
        pw.draw_text(
            self.screen,
            IMG_W * 2 + PADDING + 10,
            28,
            f"V     : {value:.2f}",
            color=(255, 255, 255),
        )
        pw.draw_text(
            self.screen,
            IMG_W * 2 + PADDING + 10,
            42,
            f"done  : {1 - pred_cont:.2f}",
            color=(255, 255, 255),
        )

        if isinstance(self.config.action_space, DiscreteSpace):
            act_dist = self.parameter.actor(self.feat)
            act_probs = act_dist.probs().numpy()[0]

            # 横にアクション後の結果を表示
            for a in range(self.config.action_space.n):
                if a > _view_action:
                    break

                pw.draw_text(
                    self.screen,
                    (IMG_W + PADDING) * a,
                    20 + IMG_H,
                    f"{worker.env.action_to_str(a)}({act_probs[a] * 100:.0f}%)",
                    color=(255, 255, 255),
                )

                action = tf.one_hot([a], self.config.action_space.n, axis=1)
                deter, prior = self.parameter.dynamics.img_step(self.stoch, self.deter, action)
                feat = tf.concat([prior["stoch"], deter], axis=1)

                next_state = self.parameter.decode(feat).sample()
                reward = self.parameter.reward(feat).mode()
                cont = self.parameter.cont(feat).mode()
                value = self.parameter.critic(feat).mode()

                n_img = next_state[0].numpy() * 255
                reward = reward.numpy()[0][0]
                cont = cont.numpy()[0][0]
                value = value.numpy()[0][0]

                x = (IMG_W + PADDING) * a
                y = 20 + IMG_H + STR_H
                pw.draw_text(
                    self.screen,
                    x,
                    y + STR_H * 0,
                    f"r={reward:5.2f}",
                    color=(255, 255, 255),
                )
                pw.draw_text(
                    self.screen,
                    x,
                    y + STR_H * 1,
                    f"d={1 - cont:5.2f}",
                    color=(255, 255, 255),
                )
                pw.draw_text(
                    self.screen,
                    x,
                    y + STR_H * 2,
                    f"V={value:5.2f}",
                    color=(255, 255, 255),
                )
                pw.draw_image_rgb_array(self.screen, x, y + STR_H * 3, n_img)

        elif isinstance(self.config.action_space, NpArraySpace):
            act_dist = self.parameter.actor(self.feat)
            action = act_dist.mean()
            deter, prior = self.parameter.dynamics.img_step(self.stoch, self.deter, action)
            feat = tf.concat([prior["stoch"], deter], axis=1)

            next_state = self.parameter.decode(feat).sample()
            reward = self.parameter.reward(feat).mode()
            cont = self.parameter.cont(feat).prob()
            value = self.parameter.critic(feat).mode()

            n_img = next_state[0].numpy() * 255
            s = f"act {action.numpy()[0]}, mean {act_dist.mean().numpy()[0]}"
            s += f", std {act_dist.stddev().numpy()[0]}"
            s += f", reward {reward.numpy()[0][0]:.5f}"
            s += f", done {(1 - cont.numpy()[0][0]) * 100:4.1f}%"
            s += f", value {value.numpy()[0][0]:.5f}"

            x = IMG_W + PADDING
            y = 20 + IMG_H + STR_H + (IMG_H + PADDING + STR_H * 3)
            pw.draw_text(self.screen, x, y + STR_H * 1, s, color=(255, 255, 255))
            pw.draw_image_rgb_array(self.screen, x, y + STR_H * 3, n_img)

        else:
            raise ValueError(self.config.action_space.stype)

        return pw.get_rgb_array(self.screen)
