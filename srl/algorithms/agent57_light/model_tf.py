import logging
from typing import Any, cast

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.base import RLTrainer
from srl.rl.functions import common
from srl.rl.models.tf.dueling_network import DuelingNetworkBlock
from srl.rl.models.tf.input_block import InputBlock

from .agent57_light import CommonInterfaceParameter, Config, RemoteMemory

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

        # --- in block
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)
        self.use_image_layer = self.in_block.use_image_layer

        # image
        if self.use_image_layer:
            self.image_block = config.image_block_config.create_block_tf()
            self.image_flatten = kl.Flatten()

        # hidden
        self.hidden_layers = []
        for i in range(len(config.hidden_layer_sizes) - 1):
            self.hidden_layers.append(
                kl.Dense(
                    config.hidden_layer_sizes[i],
                    activation=config.activation,
                    kernel_initializer="he_normal",
                )
            )

        # out
        self.enable_dueling_network = config.enable_dueling_network
        if config.enable_dueling_network:
            self.dueling_block = DuelingNetworkBlock(
                config.action_num,
                config.hidden_layer_sizes[-1],
                config.dueling_network_type,
                activation=config.activation,
            )
        else:
            self.out_layers = [
                kl.Dense(
                    config.hidden_layer_sizes[-1],
                    activation=config.activation,
                    kernel_initializer="he_normal",
                ),
                kl.Dense(
                    config.action_num,
                    kernel_initializer="truncated_normal",
                    bias_initializer="truncated_normal",
                ),
            ]

        # build
        self.build(
            (None,) + config.observation_shape,
            (None, 1),
            (None, config.action_num),
            (None, config.actor_num),
        )

    def call(self, inputs, training=False):
        state = inputs[0]
        reward_ext = inputs[1]
        reward_int = inputs[2]
        onehot_action = inputs[3]
        onehot_actor = inputs[4]

        # input
        x = self.in_block(state, training=training)
        if self.use_image_layer:
            x = self.image_block(x, training=training)
            x = self.image_flatten(x)

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

        # hidden
        for layer in self.hidden_layers:
            x = layer(x, training=training)

        # out
        if self.enable_dueling_network:
            x = self.dueling_block(x, training=training)
        else:
            for layer in self.out_layers:
                x = layer(x, training=training)

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
        if hasattr(self.in_block, "init_model_graph"):
            self.in_block.init_model_graph()
        if self.in_block.use_image_layer and hasattr(self.image_block, "init_model_graph"):
            self.image_block.init_model_graph()
        if self.enable_dueling_network and hasattr(self.dueling_block, "init_model_graph"):
            self.dueling_block.init_model_graph()
        x = [
            kl.Input(self.__in_state_shape[1:], name="state"),
            kl.Input(self.__in_reward_shape[1:], name="reward_ext"),
            kl.Input(self.__in_reward_shape[1:], name="reward_int"),
            kl.Input(self.__in_action_shape[1:], name="action"),
            kl.Input(self.__in_actor_shape[1:], name="actor"),
        ]
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x), name=name)
        model.summary(**kwargs)


# ------------------------------------------------------
# エピソード記憶部(episodic_reward)
# ------------------------------------------------------
class _EmbeddingNetwork(keras.Model):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        # input
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)

        # image
        if self.in_block.use_image_layer:
            self.image_block = config.image_block_config.create_block_tf()
            self.image_flatten = kl.Flatten()

        # emb_block
        self.emb_block = config.episodic_emb_block.create_block_tf()

        # out_block
        self.out_block = config.episodic_out_block.create_block_tf()
        self.out_block_normalize = kl.LayerNormalization()
        self.out_block_out = kl.Dense(config.action_num, activation="softmax")

        # build
        self.build((None,) + config.observation_shape)

    def _emb_block_call(self, state, training=False):
        x = self.in_block(state, training=training)
        if self.in_block.use_image_layer:
            x = self.image_block(x, training=training)
            x = self.image_flatten(x)
        return self.emb_block(x, training=training)

    def call(self, x, training=False):
        x1 = self._emb_block_call(x[0], training=training)
        x2 = self._emb_block_call(x[1], training=training)

        x = tf.concat([x1, x2], axis=1)
        x = self.out_block(x, training=training)
        x = self.out_block_normalize(x, training=training)
        x = self.out_block_out(x, training=training)
        return x

    def predict(self, state):
        return self._emb_block_call(state)

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build([self.__input_shape, self.__input_shape])

    def summary(self, name: str = "", **kwargs):
        if hasattr(self.in_block, "init_model_graph"):
            self.in_block.init_model_graph()
        if self.in_block.use_image_layer and hasattr(self.image_block, "init_model_graph"):
            self.image_block.init_model_graph()
        if hasattr(self.emb_block, "init_model_graph"):
            self.emb_block.init_model_graph()
        if hasattr(self.out_block, "init_model_graph"):
            self.out_block.init_model_graph()

        x = [
            kl.Input(shape=self.__input_shape[1:]),
            kl.Input(shape=self.__input_shape[1:]),
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
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)

        # image
        if self.in_block.use_image_layer:
            self.image_block = config.image_block_config.create_block_tf()
            self.image_flatten = kl.Flatten()

        # hidden
        self.hidden_block = config.lifelong_hidden_block.create_block_tf()
        self.hidden_normalize = kl.LayerNormalization()

        # build
        self.build((None,) + config.observation_shape)

    def call(self, x, training=False):
        x = self.in_block(x, training=training)
        if self.in_block.use_image_layer:
            x = self.image_block(x, training=training)
            x = self.image_flatten(x)
        x = self.hidden_block(x, training=training)
        x = self.hidden_normalize(x, training=training)
        return x

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
class Parameter(CommonInterfaceParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.q_ext_online = _QNetwork(self.config)
        self.q_ext_target = _QNetwork(self.config)
        self.q_int_online = _QNetwork(self.config)
        self.q_int_target = _QNetwork(self.config)
        self.emb_network = _EmbeddingNetwork(self.config)
        self.lifelong_target = _LifelongNetwork(self.config)
        self.lifelong_train = _LifelongNetwork(self.config)

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
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        self.q_ext_optimizer = keras.optimizers.Adam(learning_rate=self.config.q_ext_lr)
        self.q_int_optimizer = keras.optimizers.Adam(learning_rate=self.config.q_int_lr)
        self.q_loss = keras.losses.Huber()

        self.emb_optimizer = keras.optimizers.Adam(learning_rate=self.config.episodic_lr)
        self.emb_loss = keras.losses.MeanSquaredError()

        self.lifelong_optimizer = keras.optimizers.Adam(learning_rate=self.config.lifelong_lr)
        self.lifelong_loss = keras.losses.MeanSquaredError()

        self.beta_list = common.create_beta_list(self.config.actor_num)
        self.discount_list = common.create_discount_list(self.config.actor_num)

        self.train_count = 0
        self.sync_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):
        if self.remote_memory.length() < self.config.memory_warmup_size:
            return {}
        indices, batchs, weights = self.remote_memory.sample(self.config.batch_size, self.train_count)
        info = {}

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
        td_errors_ext, loss_ext = self._update_q(
            True,
            self.parameter.q_ext_online,
            self.q_ext_optimizer,
            rewards_ext,
            *_params,
        )
        info["loss_ext"] = loss_ext

        # --- intrinsic reward
        if self.config.enable_intrinsic_reward:
            td_errors_int, loss_int = self._update_q(
                False,
                self.parameter.q_int_online,
                self.q_int_optimizer,
                rewards_int,
                *_params,
            )

            # ----------------------------------------
            # embedding network
            # ----------------------------------------
            with tf.GradientTape() as tape:
                actions_probs = self.parameter.emb_network([states, n_states], training=True)
                emb_loss = self.emb_loss(actions_probs, onehot_actions)
                emb_loss += tf.reduce_sum(self.parameter.emb_network.losses)

            grads = tape.gradient(emb_loss, self.parameter.emb_network.trainable_variables)
            self.emb_optimizer.apply_gradients(zip(grads, self.parameter.emb_network.trainable_variables))

            # ----------------------------------------
            # lifelong network
            # ----------------------------------------
            lifelong_target_val = self.parameter.lifelong_target(states)
            with tf.GradientTape() as tape:
                lifelong_train_val = self.parameter.lifelong_train(states, training=True)
                lifelong_loss = self.lifelong_loss(lifelong_target_val, lifelong_train_val)
                lifelong_loss += tf.reduce_sum(self.parameter.lifelong_train.losses)

            grads = tape.gradient(lifelong_loss, self.parameter.lifelong_train.trainable_variables)
            self.lifelong_optimizer.apply_gradients(zip(grads, self.parameter.lifelong_train.trainable_variables))

            info["loss_int"] = loss_int
            info["emb_loss"] = emb_loss.numpy()
            info["lifelong_loss"] = lifelong_loss.numpy()
        else:
            td_errors_int = 0.0

        # --- update memory
        if self.config.disable_int_priority:
            td_errors = td_errors_ext
        else:
            batch_beta = np.array([self.beta_list[a] for a in actor_idx], np.float32)
            td_errors = td_errors_ext + batch_beta * td_errors_int
        self.remote_memory.update(indices, batchs, td_errors)

        # --- sync target
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_ext_target.set_weights(self.parameter.q_ext_online.get_weights())
            self.parameter.q_int_target.set_weights(self.parameter.q_int_online.get_weights())
            self.sync_count += 1

        self.train_count += 1
        info["sync"] = self.sync_count
        return info

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
        loss = loss.numpy()

        return td_errors, loss
