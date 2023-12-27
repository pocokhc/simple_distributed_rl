import logging
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import RLTypes
from srl.base.rl.base import RLTrainer
from srl.rl.functions import common
from srl.rl.models.tf.input_block import InputImageBlock

from .agent57_light import CommonInterfaceParameter, Config

kl = keras.layers
logger = logging.getLogger(__name__)


class _QNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()
        self.input_ext_reward = config.input_ext_reward
        self.input_int_reward = config.input_int_reward
        self.input_action = config.input_action
        if not config.enable_intrinsic_reward:
            self.input_int_reward = False

        # --- input
        self.in_img_block = None
        if config.observation_type == RLTypes.IMAGE:
            self.in_img_block = InputImageBlock(
                config.observation_shape,
                config.env_observation_type,
                enable_time_distributed_layer=True,
            )
            self.img_block = config.image_block.create_block_tf(enable_time_distributed_layer=True)
        self.flat_layer = kl.Flatten()

        # out
        self.dueling_block = config.dueling_network.create_block_tf(config.action_num)

        # build
        self.build(
            (None,) + config.observation_shape,
            (None, 1),
            (None, config.action_num),
            (None, config.actor_num),
        )

    @tf.function
    def call(self, x, training=False):
        return self._call(x, training)

    def _call(self, inputs, training=False):
        state = inputs[0]
        reward_ext = inputs[1]
        reward_int = inputs[2]
        onehot_action = inputs[3]
        onehot_actor = inputs[4]

        # input
        x = state
        if self.in_img_block is not None:
            x = self.in_img_block(x, training)
            x = self.img_block(x, training)
        x = self.flat_layer(x)

        # UVFA
        uvfa_list = [x]
        if self.input_ext_reward:
            uvfa_list.append(reward_ext)
        if self.input_int_reward:
            uvfa_list.append(reward_int)
        if self.input_action:
            uvfa_list.append(onehot_action)
        uvfa_list.append(onehot_actor)
        x = tf.concat(uvfa_list, axis=1)

        x = self.dueling_block(x, training=training)
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
        if self.in_img_block is not None:
            self.in_img_block.init_model_graph()
        self.dueling_block.init_model_graph()
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
class _EmbeddingNetwork(keras.Model):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        # input
        self.in_img_block = None
        if config.observation_type == RLTypes.IMAGE:
            self.in_img_block = InputImageBlock(config.observation_shape, config.env_observation_type)
            self.img_block = config.image_block.create_block_tf()
        self.flat_layer = kl.Flatten()

        # emb_block
        self.emb_block = config.episodic_emb_block.create_block_tf()

        # out_block
        self.out_block = config.episodic_out_block.create_block_tf()
        self.out_block_normalize = kl.LayerNormalization()
        self.out_block_out = kl.Dense(config.action_num, activation="softmax")

        # build
        self._in_shape = config.observation_shape
        self.build([(None,) + self._in_shape, (None,) + self._in_shape])
        self.loss_func = keras.losses.MeanSquaredError()

    def _emb_block_call(self, x, training=False):
        if self.in_img_block is not None:
            x = self.in_img_block(x, training=training)
            x = self.img_block(x, training=training)
        x = self.flat_layer(x)
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

    def summary(self, name: str = "", **kwargs):
        if self.in_img_block is not None:
            self.in_img_block.init_model_graph()
            self.img_block.init_model_graph()
        self.emb_block.init_model_graph()
        self.out_block.init_model_graph()

        x = [
            kl.Input(shape=self._in_shape),
            kl.Input(shape=self._in_shape),
        ]
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x), name=name)
        model.summary(**kwargs)


# ------------------------------------------------------
# 生涯記憶部(life long novelty module)
# ------------------------------------------------------
class _LifelongNetwork(keras.Model):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        # input
        self.in_img_block = None
        if config.observation_type == RLTypes.IMAGE:
            self.in_img_block = InputImageBlock(config.observation_shape, config.env_observation_type)
            self.img_block = config.image_block.create_block_tf()
        self.flat_layer = kl.Flatten()

        # hidden
        self.hidden_block = config.lifelong_hidden_block.create_block_tf()
        self.hidden_normalize = kl.LayerNormalization()

        # build
        self._in_shape = config.observation_shape
        self.build((None,) + self._in_shape)
        self.loss_func = keras.losses.MeanSquaredError()

    def call(self, x, training=False):
        if self.in_img_block is not None:
            x = self.in_img_block(x, training)
            x = self.img_block(x, training)
        x = self.flat_layer(x)
        x = self.hidden_block(x, training=training)
        x = self.hidden_normalize(x, training=training)
        return x

    @tf.function
    def compute_train_loss(self, state, target_val):
        val = self(state, training=True)
        loss = self.loss_func(target_val, val)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss

    def summary(self, name: str = "", **kwargs):
        if self.in_img_block is not None:
            self.in_img_block.init_model_graph()
            self.img_block.init_model_graph()
        self.hidden_block.init_model_graph()

        x = kl.Input(shape=self._in_shape)
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x), name=name)
        model.summary(**kwargs)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(CommonInterfaceParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.q_ext_online = _QNetwork(self.config)
        self.q_ext_target = _QNetwork(self.config)
        self.q_int_online = _QNetwork(self.config)
        self.q_int_target = _QNetwork(self.config)
        self.emb_network = _EmbeddingNetwork(self.config)
        self.lifelong_target = _LifelongNetwork(self.config)
        self.lifelong_train = _LifelongNetwork(self.config)

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
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_sch_ext = self.config.lr_ext.create_schedulers()
        self.lr_sch_int = self.config.lr_int.create_schedulers()
        self.lr_sch_emb = self.config.episodic_lr.create_schedulers()
        self.lr_sch_ll = self.config.lifelong_lr.create_schedulers()

        self.q_ext_optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch_ext.get_rate())
        self.q_int_optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch_int.get_rate())
        self.q_loss = keras.losses.Huber()

        self.emb_optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch_emb.get_rate())
        self.lifelong_optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch_ll.get_rate())

        self.beta_list = common.create_beta_list(self.config.actor_num)
        self.discount_list = common.create_discount_list(self.config.actor_num)

        self.sync_count = 0

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        indices, batchs, weights = self.memory.sample(self.batch_size, self.train_count)

        _info = {}
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
        td_errors_ext, _loss = self._update_q(
            True,
            self.parameter.q_ext_online,
            self.q_ext_optimizer,
            self.lr_sch_ext,
            rewards_ext,
            *_params,
        )
        _info["ext_loss"] = _loss

        # --- intrinsic reward
        if self.config.enable_intrinsic_reward:
            td_errors_int, _loss = self._update_q(
                False,
                self.parameter.q_int_online,
                self.q_int_optimizer,
                self.lr_sch_int,
                rewards_int,
                *_params,
            )
            _info["int_loss"] = _loss

            # ----------------------------------------
            # embedding network
            # ----------------------------------------
            with tf.GradientTape() as tape:
                emb_loss = self.parameter.emb_network.compute_train_loss(states, n_states, onehot_actions)
            grads = tape.gradient(emb_loss, self.parameter.emb_network.trainable_variables)
            self.emb_optimizer.apply_gradients(zip(grads, self.parameter.emb_network.trainable_variables))
            _info["emb_loss"] = emb_loss.numpy()

            if self.lr_sch_emb.update(self.train_count):
                self.emb_optimizer.learning_rate = self.lr_sch_emb.get_rate()

            # ----------------------------------------
            # lifelong network
            # ----------------------------------------
            lifelong_target_val = self.parameter.lifelong_target(states)
            with tf.GradientTape() as tape:
                lifelong_loss = self.parameter.lifelong_train.compute_train_loss(states, lifelong_target_val)
            grads = tape.gradient(lifelong_loss, self.parameter.lifelong_train.trainable_variables)
            self.lifelong_optimizer.apply_gradients(zip(grads, self.parameter.lifelong_train.trainable_variables))
            _info["lifelong_loss"] = lifelong_loss.numpy()

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
        self.memory.update((indices, batchs, priorities))

        # --- sync target
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_ext_target.set_weights(self.parameter.q_ext_online.get_weights())
            self.parameter.q_int_target.set_weights(self.parameter.q_int_online.get_weights())
            self.sync_count += 1
        _info["sync"] = self.sync_count

        self.train_info = _info
        self.train_count += 1

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

        return td_errors, loss.numpy()
