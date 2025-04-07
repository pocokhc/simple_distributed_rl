import logging
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.trainer import RLTrainer
from srl.rl import functions as funcs
from srl.rl.tf.model import KerasModelAddedSummary

from .agent57_light import CommonInterfaceParameter, Config, Memory

kl = keras.layers
logger = logging.getLogger(__name__)


class QNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config):
        super().__init__()
        self.input_ext_reward = config.input_ext_reward
        self.input_int_reward = config.input_int_reward
        self.input_action = config.input_action
        if not config.enable_intrinsic_reward:
            self.input_int_reward = False

        if config.observation_space.is_value():
            self.in_block = config.input_value_block.create_tf_block(config.observation_space)
        elif config.observation_space.is_image():
            self.in_block = config.input_image_block.create_tf_block(config.observation_space)
        else:
            raise ValueError(config.observation_space)

        self.concat_layer = kl.Concatenate(axis=-1)
        self.hidden_block = config.hidden_block.create_tf_block(config.action_space.n)

        # build
        self(
            [
                np.zeros((1,) + config.observation_space.shape),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                np.zeros((1, config.action_space.n)),
                np.zeros((1, config.actor_num)),
            ]
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
        state = self.in_block(state, training=training)

        # UVFA
        uvfa_list = [state]
        if self.input_ext_reward:
            uvfa_list.append(reward_ext)
        if self.input_int_reward:
            uvfa_list.append(reward_int)
        if self.input_action:
            uvfa_list.append(onehot_action)
        uvfa_list.append(onehot_actor)
        x = self.concat_layer(uvfa_list)

        x = self.hidden_block(x, training=training)
        return x


# ------------------------------------------------------
# エピソード記憶部(episodic_reward)
# ------------------------------------------------------
class EmbeddingNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        if config.observation_space.is_value():
            self.in_block = config.input_value_block.create_tf_block(config.observation_space)
        elif config.observation_space.is_image():
            self.in_block = config.input_image_block.create_tf_block(config.observation_space)
        else:
            raise ValueError(config.observation_space)

        self.emb_block = config.episodic_emb_block.create_tf_block()

        self.concat_layer = kl.Concatenate(axis=-1)
        self.out_block = config.episodic_out_block.create_tf_block()
        self.out_block_normalize = kl.LayerNormalization()
        self.out_block_out = kl.Dense(config.action_space.n, activation="softmax")

        # build
        self(
            [
                np.zeros((1,) + config.observation_space.shape),
                np.zeros((1,) + config.observation_space.shape),
            ]
        )
        self.loss_func = keras.losses.MeanSquaredError()

    def _emb_block_call(self, x, training=False):
        x = self.in_block(x, training=training)
        return self.emb_block(x, training=training)

    def call(self, x, training=False):
        x1 = self._emb_block_call(x[0], training=training)
        x2 = self._emb_block_call(x[1], training=training)

        x = self.concat_layer([x1, x2])
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
class LifelongNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        if config.observation_space.is_value():
            self.in_block = config.input_value_block.create_tf_block(config.observation_space)
        elif config.observation_space.is_image():
            self.in_block = config.input_image_block.create_tf_block(config.observation_space)
        else:
            raise ValueError(config.observation_space)

        self.hidden_block = config.lifelong_hidden_block.create_tf_block()
        self.hidden_normalize = kl.LayerNormalization()

        # build
        self(np.zeros((1,) + config.observation_space.shape))
        self.loss_func = keras.losses.MeanSquaredError()

    def call(self, x, training=False):
        x = self.in_block(x, training=training)
        x = self.hidden_block(x, training=training)
        x = self.hidden_normalize(x, training=training)
        return x

    @tf.function
    def compute_train_loss(self, state, target_val):
        val = self(state, training=True)
        loss = self.loss_func(target_val, val)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss


class Parameter(CommonInterfaceParameter):
    def setup(self):
        super().setup()
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


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self):
        self.q_ext_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr_ext_scheduler.apply_tf_scheduler(self.config.lr_ext))
        self.q_int_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr_int_scheduler.apply_tf_scheduler(self.config.lr_int))
        self.q_loss = keras.losses.Huber()

        self.emb_optimizer = keras.optimizers.Adam(learning_rate=self.config.episodic_lr_scheduler.apply_tf_scheduler(self.config.episodic_lr))
        self.lifelong_optimizer = keras.optimizers.Adam(learning_rate=self.config.lifelong_lr_scheduler.apply_tf_scheduler(self.config.lifelong_lr))

        self.beta_list = funcs.create_beta_list(self.config.actor_num)
        self.discount_list = funcs.create_discount_list(self.config.actor_num)

        self.sync_count = 0

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return
        batches, weights, update_args = batches

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
        ) = self.parameter.change_batches_format(batches)

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
        td_errors_ext, ext_loss = self._update_q(
            True,
            self.parameter.q_ext_online,
            self.q_ext_optimizer,
            rewards_ext,
            *_params,
        )
        self.info["ext_loss"] = ext_loss.numpy()

        # --- intrinsic reward
        if self.config.enable_intrinsic_reward:
            td_errors_int, int_loss = self._update_q(
                False,
                self.parameter.q_int_online,
                self.q_int_optimizer,
                rewards_int,
                *_params,
            )
            self.info["int_loss"] = int_loss.numpy()

            # ----------------------------------------
            # embedding network
            # ----------------------------------------
            with tf.GradientTape() as tape:
                emb_loss = self.parameter.emb_network.compute_train_loss(states, n_states, onehot_actions)
            grads = tape.gradient(emb_loss, self.parameter.emb_network.trainable_variables)
            self.emb_optimizer.apply_gradients(zip(grads, self.parameter.emb_network.trainable_variables))
            self.info["emb_loss"] = emb_loss.numpy()

            # ----------------------------------------
            # lifelong network
            # ----------------------------------------
            lifelong_target_val = self.parameter.lifelong_target(states)
            with tf.GradientTape() as tape:
                lifelong_loss = self.parameter.lifelong_train.compute_train_loss(states, lifelong_target_val)
            grads = tape.gradient(lifelong_loss, self.parameter.lifelong_train.trainable_variables)
            self.lifelong_optimizer.apply_gradients(zip(grads, self.parameter.lifelong_train.trainable_variables))
            self.info["lifelong_loss"] = lifelong_loss.numpy()

        else:
            td_errors_int = 0.0

        # --- update memory
        if self.config.disable_int_priority:
            priorities = np.abs(td_errors_ext)
        else:
            batch_beta = np.array([self.beta_list[a] for a in actor_idx], np.float32)
            priorities = np.abs(td_errors_ext + batch_beta * td_errors_int)
        self.memory.update(update_args, priorities, self.train_count)

        # --- sync target
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_ext_target.set_weights(self.parameter.q_ext_online.get_weights())
            self.parameter.q_int_target.set_weights(self.parameter.q_int_online.get_weights())
            self.sync_count += 1

        self.info["sync"] = self.sync_count
        self.train_count += 1

    def _update_q(
        self,
        is_ext: bool,
        model_q_online,
        optimizer,
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

        td_errors = target_q - q.numpy()

        return td_errors, loss
