from typing import Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.base import RLTrainer
from srl.rl.functions.common import create_beta_list, create_discount_list
from srl.rl.models.tf.input_block import InputBlock

from .agent57 import CommonInterfaceParameter, Config, RemoteMemory

kl = keras.layers


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()
        self.input_ext_reward = config.input_ext_reward
        self.input_int_reward = config.input_int_reward
        self.input_action = config.input_action
        if not config.enable_intrinsic_reward:
            self.input_int_reward = False

        # --- in block
        self.in_block = InputBlock(
            config.observation_shape,
            config.env_observation_type,
            enable_time_distributed_layer=True,
        )
        self.use_image_layer = self.in_block.use_image_layer

        # image
        if self.use_image_layer:
            self.image_block = config.image_block.create_block_tf(enable_time_distributed_layer=True)
            self.image_flatten = kl.TimeDistributed(kl.Flatten())

        # --- lstm
        self.lstm_layer = kl.LSTM(config.lstm_units, return_sequences=True, return_state=True)

        # out
        self.dueling_block = config.dueling_network.create_block_tf(
            config.action_num,
            enable_time_distributed_layer=True,
        )

        # build
        self.build(
            (
                None,
                config.sequence_length,
            )
            + config.observation_shape,
            (None, config.sequence_length, 1),
            (None, config.sequence_length, config.action_num),
            (None, config.sequence_length, config.actor_num),
        )

    @tf.function()
    def call(self, x, hidden_states=None, training=False):
        return self._call(x, hidden_states, training)

    def _call(self, inputs, hidden_states=None, training=False):
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
        x = tf.concat(uvfa_list, axis=2)

        # lstm
        x, h, c = self.lstm_layer(x, initial_state=hidden_states, training=training)

        x = self.dueling_block(x, training=training)
        return x, [h, c]

    def get_initial_state(self, batch_size=1):
        return self.lstm_layer.cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

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
        self.in_block.init_model_graph()
        if self.in_block.use_image_layer:
            self.image_block.init_model_graph()
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
    def __init__(self, config: Config):
        super().__init__()

        # input
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)

        # image
        if self.in_block.use_image_layer:
            self.image_block = config.image_block.create_block_tf(enable_time_distributed_layer=False)
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

    @tf.function()
    def call(self, x, training=False):
        return self._call(x, training)

    def _call(self, x, training=False):
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
        self.in_block.init_model_graph()
        if self.in_block.use_image_layer:
            self.image_block.init_model_graph()
        self.emb_block.init_model_graph()
        self.out_block.init_model_graph()

        x = [
            kl.Input(shape=self.__input_shape[1:]),
            kl.Input(shape=self.__input_shape[1:]),
        ]
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self._call(x), name=name)
        model.summary(**kwargs)


# ------------------------------------------------------
# 生涯記憶部(life long novelty module)
# ------------------------------------------------------
class _LifelongNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        # input
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)

        # image
        if self.in_block.use_image_layer:
            self.image_block = config.image_block.create_block_tf(enable_time_distributed_layer=False)
            self.image_flatten = kl.Flatten()

        # hidden
        self.hidden_block = config.lifelong_hidden_block.create_block_tf()
        self.hidden_normalize = kl.LayerNormalization()

        # build
        self.build((None,) + config.observation_shape)

    @tf.function()
    def call(self, x, training=False):
        return self._call(x, training)

    def _call(self, x, training=False):
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
        self.in_block.init_model_graph()
        if self.in_block.use_image_layer:
            self.image_block.init_model_graph()
        self.hidden_block.init_model_graph()

        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self._call(x), name=name)
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

    # ------------

    def get_initial_hidden_state_q_ext(self) -> Any:
        return self.q_ext_online.get_initial_state()

    def get_initial_hidden_state_q_int(self) -> Any:
        return self.q_int_online.get_initial_state()

    def predict_q_ext_online(self, x, hidden_state) -> Tuple[np.ndarray, Any]:
        q, h = self.q_ext_online(x, hidden_state)
        return q.numpy(), h

    def predict_q_int_online(self, x, hidden_state) -> Tuple[np.ndarray, Any]:
        q, h = self.q_int_online(x, hidden_state)
        return q.numpy(), h

    def predict_emb(self, x) -> np.ndarray:
        return self.emb_network.predict(x).numpy()

    def predict_lifelong_target(self, x) -> np.ndarray:
        return self.lifelong_target(x).numpy()

    def predict_lifelong_train(self, x) -> np.ndarray:
        return self.lifelong_train(x).numpy()

    def convert_numpy_from_hidden_state(self, h):
        return [h[0][0], h[1][0]]


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter
        self.remote_memory: RemoteMemory = self.remote_memory

        self.lr_sch_ext = self.config.lr_ext.create_schedulers()
        self.lr_sch_int = self.config.lr_int.create_schedulers()
        self.lr_sch_emb = self.config.episodic_lr.create_schedulers()
        self.lr_sch_ll = self.config.lifelong_lr.create_schedulers()

        self.q_ext_optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch_ext.get_rate(0))
        self.q_int_optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch_int.get_rate(0))
        self.q_loss = keras.losses.Huber()

        self.emb_optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch_emb.get_rate(0))
        self.emb_loss = keras.losses.MeanSquaredError()

        self.lifelong_optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch_ll.get_rate(0))
        self.lifelong_loss = keras.losses.MeanSquaredError()

        self.beta_list = create_beta_list(self.config.actor_num)
        self.discount_list = create_discount_list(self.config.actor_num)

        self.sync_count = 0

    def train_on_batchs(self, memory_sample_return) -> None:
        indices, batchs, weights = memory_sample_return

        (
            burnin_states,
            burnin_rewards_ext,
            burnin_rewards_int,
            burnin_actions_onehot,
            burnin_actor_onehot,
            instep_states,
            instep_rewards_ext,
            instep_rewards_int,
            instep_actions_onehot,
            instep_actor_onehot,
            step_rewards_ext,
            step_rewards_int,
            step_actions_onehot,
            step_dones,
            inv_act_idx1,
            inv_act_idx2,
            inv_act_idx3,
            hidden_states_ext,
            hidden_states_int,
            discount_list,
            beta_list,
            weights,
        ) = self.parameter.change_batchs_format(batchs, weights)

        # hidden_states
        states_h_ext = [h[0] for h in hidden_states_ext]
        states_c_ext = [h[1] for h in hidden_states_ext]
        states_h_int = [h[0] for h in hidden_states_int]
        states_c_int = [h[1] for h in hidden_states_int]
        hidden_states_ext = [tf.stack(states_h_ext), tf.stack(states_c_ext)]
        hidden_states_int = [tf.stack(states_h_int), tf.stack(states_c_int)]
        hidden_states_ext_t = hidden_states_ext
        hidden_states_int_t = hidden_states_int

        _params = [
            [
                burnin_states,
                burnin_rewards_ext,
                burnin_rewards_int,
                burnin_actions_onehot,
                burnin_actor_onehot,
            ],
            [
                instep_states,
                instep_rewards_ext,
                instep_rewards_int,
                instep_actions_onehot,
                instep_actor_onehot,
            ],
            step_actions_onehot,
            step_dones,
            inv_act_idx1,
            inv_act_idx2,
            inv_act_idx3,
            discount_list,
            weights,
        ]
        td_error_ext, _loss, _lr = self._train_q(
            self.parameter.q_ext_online,
            self.parameter.q_ext_target,
            self.q_ext_optimizer,
            self.lr_sch_ext,
            step_rewards_ext,
            hidden_states_ext,
            hidden_states_ext_t,
            *_params,
        )
        _info = {"ext_loss": _loss, "ext_lr": _lr}

        if self.config.enable_intrinsic_reward:
            td_error_int, _loss, _lr = self._train_q(
                self.parameter.q_int_online,
                self.parameter.q_int_target,
                self.q_int_optimizer,
                self.lr_sch_int,
                step_rewards_int,
                hidden_states_int,
                hidden_states_int_t,
                *_params,
            )
            _info["int_loss"] = _loss
            _info["int_lr"] = _lr

            # embedding lifelong (batch, step, x) -> (batch, x)
            one_states = instep_states[:, 0, ...]
            one_n_states = instep_states[:, 1, ...]
            one_actions_onehot = instep_actions_onehot[:, 0, :].astype(np.float32)

            # ----------------------------------------
            # embedding network
            # ----------------------------------------
            with tf.GradientTape() as tape:
                actions_probs = self.parameter.emb_network([one_states, one_n_states], training=True)
                emb_loss = self.emb_loss(actions_probs, one_actions_onehot)
                emb_loss += tf.reduce_sum(self.parameter.emb_network.losses)

            grads = tape.gradient(emb_loss, self.parameter.emb_network.trainable_variables)
            self.emb_optimizer.apply_gradients(zip(grads, self.parameter.emb_network.trainable_variables))
            _info["emb_loss"] = emb_loss.numpy()
            lr = self.lr_sch_emb.get_rate(self.train_count)
            self.emb_optimizer.learning_rate = lr
            _info["emb_lr"] = lr

            # ----------------------------------------
            # lifelong network
            # ----------------------------------------
            lifelong_target_val = self.parameter.lifelong_target(one_states)
            with tf.GradientTape() as tape:
                lifelong_train_val = self.parameter.lifelong_train(one_states, training=True)
                lifelong_loss = self.lifelong_loss(lifelong_target_val, lifelong_train_val)
                lifelong_loss += tf.reduce_sum(self.parameter.lifelong_train.losses)

            grads = tape.gradient(lifelong_loss, self.parameter.lifelong_train.trainable_variables)
            self.lifelong_optimizer.apply_gradients(zip(grads, self.parameter.lifelong_train.trainable_variables))
            _info["lifelong_loss"] = lifelong_loss.numpy()
            lr = self.lr_sch_ll.get_rate(self.train_count)
            self.lifelong_optimizer.learning_rate = lr
            _info["lifelong_lr"] = lr

        else:
            td_error_int = 0

        if self.config.disable_int_priority:
            td_errors = td_error_ext
        else:
            td_errors = td_error_ext + beta_list * td_error_int


        # --- sync target
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_ext_target.set_weights(self.parameter.q_ext_online.get_weights())
            self.parameter.q_int_target.set_weights(self.parameter.q_int_online.get_weights())
            self.sync_count += 1

        _info["sync"] = self.sync_count
        self.train_count += 1
        self.train_info = _info

    def _train_q(
        self,
        model_q_online,
        model_q_target,
        optimizer,
        lr_sch,
        step_rewards,
        hidden_states,
        hidden_states_t,
        #
        in_burnin,
        in_steps,
        step_actions_onehot,
        step_dones,
        inv_act_idx1,
        inv_act_idx2,
        inv_act_idx3,
        discounts,
        weights,
    ):
        # burnin
        if self.config.burnin > 0:
            _, hidden_states = model_q_online(in_burnin, hidden_states)
            _, hidden_states_t = model_q_target(in_burnin, hidden_states_t)

        # targetQ
        q_target, _ = model_q_target(in_steps, hidden_states_t)
        q_target = q_target.numpy()

        with tf.GradientTape() as tape:
            # (batch, step, action)
            q, _ = model_q_online(in_steps, hidden_states, training=True)
            action_q = tf.reduce_sum(q[:, :-1, :] * step_actions_onehot, axis=2)

            target_q = self.parameter.calc_target_q(
                tf.stop_gradient(q).numpy(),
                q_target,
                tf.stop_gradient(action_q).numpy(),
                step_rewards,
                step_actions_onehot,
                step_dones,
                inv_act_idx1,
                inv_act_idx2,
                inv_act_idx3,
                discounts,
            )

            # (batch, step) -> (step, batch)
            action_q = tf.transpose(action_q, perm=[1, 0])

            # --- update Q
            loss = self.q_loss(target_q * weights, action_q * weights)
            loss += tf.nn.scale_regularization_loss(model_q_online.losses)

        grads = tape.gradient(loss, model_q_online.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_q_online.trainable_variables))
        lr = lr_sch.get_rate(self.train_count)
        optimizer.learning_rate = lr

        td_errors = np.mean(action_q - target_q, axis=0)
        return td_errors, loss.numpy(), lr
