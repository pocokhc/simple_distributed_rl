import logging
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import InfoType
from srl.base.rl.trainer import RLTrainer
from srl.rl.functions import common
from srl.rl.models.tf import helper
from srl.rl.models.tf.blocks.input_block import create_in_block_out_value
from srl.rl.schedulers.scheduler import SchedulerConfig

from .agent57_light import CommonInterfaceParameter, Config

kl = keras.layers
logger = logging.getLogger(__name__)


class QNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()
        self.input_ext_reward = config.input_ext_reward
        self.input_int_reward = config.input_int_reward
        self.input_action = config.input_action
        if not config.enable_intrinsic_reward:
            self.input_int_reward = False

        # --- input
        self.input_block = create_in_block_out_value(
            config.input_value_block,
            config.input_image_block,
            config.observation_space,
            enable_rnn=False,
        )

        # out
        self.hidden_block = config.hidden_block.create_block_tf(
            config.action_space.n,
            enable_rnn=False,
        )

        # build
        self.build(
            helper.create_batch_shape(config.observation_space.shape, (None,)),
            (None, 1),
            (None, config.action_space.n),
            (None, config.actor_num),
        )

    @tf.function
    def call(self, x, training=False):
        return self._call(x, training=training)

    def _call(self, inputs, training=False):
        state = inputs[0]
        reward_ext = inputs[1]
        reward_int = inputs[2]
        onehot_action = inputs[3]
        onehot_actor = inputs[4]

        # input
        state = self.input_block(state, training=training)

        # UVFA
        uvfa_list = [state]
        if self.input_ext_reward:
            uvfa_list.append(reward_ext)
        if self.input_int_reward:
            uvfa_list.append(reward_int)
        if self.input_action:
            uvfa_list.append(onehot_action)
        uvfa_list.append(onehot_actor)
        x = tf.concat(uvfa_list, axis=1)

        x = self.hidden_block(x, training=training)
        return x

    def build(
        self,
        in_state_shape,
        in_reward_shape,
        in_action_shape,
        in_actor_shape,
    ):
        self.__in_state_shape = in_state_shape
        self.__in_reward_shape = in_reward_shape
        self.__in_action_shape = in_action_shape
        self.__in_actor_shape = in_actor_shape
        super().build(
            [
                self.__in_state_shape,
                self.__in_reward_shape,
                self.__in_reward_shape,
                self.__in_action_shape,
                self.__in_actor_shape,
            ]
        )

    def summary(self, name="", **kwargs):
        x = [
            kl.Input(self.__in_state_shape[1:], name="state"),
            kl.Input(self.__in_reward_shape[1:], name="reward_ext"),
            kl.Input(self.__in_reward_shape[1:], name="reward_int"),
            kl.Input(self.__in_action_shape[1:], name="action"),
            kl.Input(self.__in_actor_shape[1:], name="actor"),
        ]
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self._call(x), name=name)
        model.summary(**kwargs)


# ------------------------------------------------------
# エピソード記憶部(episodic_reward)
# ------------------------------------------------------
class EmbeddingNetwork(keras.Model):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        # input
        self.input_block = create_in_block_out_value(
            config.input_value_block,
            config.input_image_block,
            config.observation_space,
        )

        # emb_block
        self.emb_block = config.episodic_emb_block.create_block_tf()

        # out_block
        self.out_block = config.episodic_out_block.create_block_tf()
        self.out_block_normalize = kl.LayerNormalization()
        self.out_block_out = kl.Dense(config.action_space.n, activation="softmax")

        # build
        self._in_shape = config.observation_space.shape
        self.build([(None,) + self._in_shape, (None,) + self._in_shape])
        self.loss_func = keras.losses.MeanSquaredError()

    def _emb_block_call(self, x, training=False):
        x = self.input_block(x, training=training)
        return self.emb_block(x, training=training)

    def call(self, x, training=False):
        x1 = self._emb_block_call(x[0], training=training)
        x2 = self._emb_block_call(x[1], training=training)

        x = tf.concat([x1, x2], axis=1)
        x = self.out_block(x, training=training)
        x = self.out_block_normalize(x, training=training)
        x = self.out_block_out(x, training=training)
        return x

    @tf.function
    def compute_train_loss(self, state, n_state, onehot_action):
        actions_probs = self([state, n_state], training=True)
        loss = self.loss_func(actions_probs, onehot_action)
        loss += tf.reduce_sum(self.losses)
        return loss

    def predict(self, state):
        return self._emb_block_call(state)


# ------------------------------------------------------
# 生涯記憶部(life long novelty module)
# ------------------------------------------------------
class LifelongNetwork(keras.Model):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        # input
        self.input_block = create_in_block_out_value(
            config.input_value_block,
            config.input_image_block,
            config.observation_space,
        )

        # hidden
        self.hidden_block = config.lifelong_hidden_block.create_block_tf()
        self.hidden_normalize = kl.LayerNormalization()

        # build
        self._in_shape = config.observation_space.shape
        self.build((None,) + self._in_shape)
        self.loss_func = keras.losses.MeanSquaredError()

    def call(self, x, training=False):
        x = self.input_block(x, training=training)
        x = self.hidden_block(x, training=training)
        x = self.hidden_normalize(x, training=training)
        return x

    @tf.function
    def compute_train_loss(self, state, target_val):
        val = self(state, training=True)
        loss = self.loss_func(target_val, val)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(CommonInterfaceParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.q_ext_online = QNetwork(self.config)
        self.q_ext_target = QNetwork(self.config)
        self.q_int_online = QNetwork(self.config)
        self.q_int_target = QNetwork(self.config)
        self.emb_network = EmbeddingNetwork(self.config)
        self.lifelong_target = LifelongNetwork(self.config)
        self.lifelong_train = LifelongNetwork(self.config)

        self.q_ext_target.set_weights(self.q_ext_online.get_weights())
        self.q_int_target.set_weights(self.q_int_online.get_weights())

    def call_restore(self, data: Any, **kwargs) -> None:
        self.q_ext_online.set_weights(data[0])
        self.q_ext_target.set_weights(data[0])
        self.q_int_online.set_weights(data[1])
        self.q_int_target.set_weights(data[1])
        self.emb_network.set_weights(data[2])
        self.lifelong_target.set_weights(data[3])
        self.lifelong_train.set_weights(data[4])

    def call_backup(self, **kwargs):
        d = [
            self.q_ext_online.get_weights(),
            self.q_int_online.get_weights(),
            self.emb_network.get_weights(),
            self.lifelong_target.get_weights(),
            self.lifelong_train.get_weights(),
        ]
        return d

    def summary(self, **kwargs):
        self.q_ext_online.summary(**kwargs)
        self.emb_network.summary(**kwargs)
        self.lifelong_target.summary(**kwargs)

    def predict_q_ext_online(self, x) -> np.ndarray:
        return self.q_ext_online(x).numpy()

    def predict_q_ext_target(self, x) -> np.ndarray:
        return self.q_ext_target(x).numpy()

    def predict_q_int_online(self, x) -> np.ndarray:
        return self.q_int_online(x).numpy()

    def predict_q_int_target(self, x) -> np.ndarray:
        return self.q_int_target(x).numpy()

    def predict_emb(self, x) -> np.ndarray:
        return self.emb_network.predict(x).numpy()

    def predict_lifelong_target(self, x) -> np.ndarray:
        return self.lifelong_target(x).numpy()

    def predict_lifelong_train(self, x) -> np.ndarray:
        return self.lifelong_train(x).numpy()


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_sch_ext = SchedulerConfig.create_scheduler(self.config.lr_ext)
        self.lr_sch_int = SchedulerConfig.create_scheduler(self.config.lr_int)
        self.lr_sch_emb = SchedulerConfig.create_scheduler(self.config.episodic_lr)
        self.lr_sch_ll = SchedulerConfig.create_scheduler(self.config.lifelong_lr)

        self.q_ext_optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch_ext.get_rate())
        self.q_int_optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch_int.get_rate())
        self.q_loss = keras.losses.Huber()

        self.emb_optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch_emb.get_rate())
        self.lifelong_optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch_ll.get_rate())

        self.beta_list = common.create_beta_list(self.config.actor_num)
        self.discount_list = common.create_discount_list(self.config.actor_num)

        self.sync_count = 0
        self.ext_loss = None
        self.int_loss = None
        self.emb_loss = None
        self.lifelong_loss = None

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        indices, batchs, weights = self.memory.sample(self.batch_size, self.train_count)

        (
            states,
            n_states,
            onehot_actions,
            next_invalid_actions,
            next_invalid_actions_idx,
            rewards_ext,
            rewards_int,
            dones,
            prev_onehot_actions,
            prev_rewards_ext,
            prev_rewards_int,
            actor_idx,
            actor_idx_onehot,
        ) = self.parameter.change_batchs_format(batchs)

        batch_discount = np.array([self.discount_list[a] for a in actor_idx], np.float32)

        # --- common params
        _params = [
            n_states,
            rewards_ext,
            rewards_int,
            onehot_actions,
            actor_idx_onehot,
            next_invalid_actions_idx,
            next_invalid_actions,
            dones,
            batch_discount,
            #
            states,
            prev_rewards_ext,
            prev_rewards_int,
            prev_onehot_actions,
            weights,
        ]

        # --- update ext q
        td_errors_ext, self.ext_loss = self._update_q(
            True,
            self.parameter.q_ext_online,
            self.q_ext_optimizer,
            self.lr_sch_ext,
            rewards_ext,
            *_params,
        )

        # --- intrinsic reward
        if self.config.enable_intrinsic_reward:
            td_errors_int, self.int_loss = self._update_q(
                False,
                self.parameter.q_int_online,
                self.q_int_optimizer,
                self.lr_sch_int,
                rewards_int,
                *_params,
            )

            # ----------------------------------------
            # embedding network
            # ----------------------------------------
            with tf.GradientTape() as tape:
                self.emb_loss = self.parameter.emb_network.compute_train_loss(states, n_states, onehot_actions)
            grads = tape.gradient(self.emb_loss, self.parameter.emb_network.trainable_variables)
            self.emb_optimizer.apply_gradients(zip(grads, self.parameter.emb_network.trainable_variables))

            if self.lr_sch_emb.update(self.train_count):
                self.emb_optimizer.learning_rate = self.lr_sch_emb.get_rate()

            # ----------------------------------------
            # lifelong network
            # ----------------------------------------
            lifelong_target_val = self.parameter.lifelong_target(states)
            with tf.GradientTape() as tape:
                self.lifelong_loss = self.parameter.lifelong_train.compute_train_loss(states, lifelong_target_val)
            grads = tape.gradient(self.lifelong_loss, self.parameter.lifelong_train.trainable_variables)
            self.lifelong_optimizer.apply_gradients(zip(grads, self.parameter.lifelong_train.trainable_variables))

            if self.lr_sch_ll.update(self.train_count):
                self.lifelong_optimizer.learning_rate = self.lr_sch_ll.get_rate()

        else:
            td_errors_int = 0.0

        # --- update memory
        if self.config.disable_int_priority:
            priorities = np.abs(td_errors_ext)
        else:
            batch_beta = np.array([self.beta_list[a] for a in actor_idx], np.float32)
            priorities = np.abs(td_errors_ext + batch_beta * td_errors_int)
        self.memory.update(indices, batchs, priorities)

        # --- sync target
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_ext_target.set_weights(self.parameter.q_ext_online.get_weights())
            self.parameter.q_int_target.set_weights(self.parameter.q_int_online.get_weights())
            self.sync_count += 1

        self.train_count += 1

    def create_info(self) -> InfoType:
        d = {"sync": self.sync_count}
        if self.ext_loss is not None:
            d["ext_loss"] = self.ext_loss.numpy()
        if self.int_loss is not None:
            d["int_loss"] = self.int_loss.numpy()
        if self.emb_loss is not None:
            d["emb_loss"] = self.emb_loss.numpy()
        if self.lifelong_loss is not None:
            d["lifelong_loss"] = self.lifelong_loss.numpy()
        self.ext_loss = None
        self.int_loss = None
        self.emb_loss = None
        self.lifelong_loss = None
        return d

    def _update_q(
        self,
        is_ext: bool,
        model_q_online,
        optimizer,
        lr_sch,
        rewards,  # (batch, 1)
        #
        n_states,
        rewards_ext,  # (batch, 1)
        rewards_int,  # (batch, 1)
        actions_onehot,
        actor_idx_onehot,
        next_invalid_actions_idx,
        next_invalid_actions,
        dones,  # (batch)
        batch_discount,  # (batch)
        #
        states,
        prev_rewards_ext,  # (batch, 1)
        prev_rewards_int,  # (batch, 1)
        prev_actions_onehot,
        weights,  # (batch)
    ):
        target_q = self.parameter.calc_target_q(
            is_ext,
            rewards.reshape(-1),
            #
            n_states,
            rewards_ext,
            rewards_int,
            actions_onehot,
            actor_idx_onehot,
            next_invalid_actions_idx,
            next_invalid_actions,
            dones,
            batch_discount,
        )

        with tf.GradientTape() as tape:
            q = model_q_online(
                [
                    states,
                    prev_rewards_ext,
                    prev_rewards_int,
                    prev_actions_onehot,
                    actor_idx_onehot,
                ],
                training=True,
            )
            q = tf.reduce_sum(q * actions_onehot, axis=1)  # (batch, shape) -> (batch)
            loss = self.q_loss(target_q * weights, q * weights)
            loss += tf.reduce_sum(model_q_online.losses)

        grads = tape.gradient(loss, model_q_online.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_q_online.trainable_variables))

        if lr_sch.update(self.train_count):
            optimizer.learning_rate = lr_sch.get_rate()

        td_errors = target_q - q.numpy()

        return td_errors, loss
