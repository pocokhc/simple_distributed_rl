from typing import Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.trainer import RLTrainer
from srl.rl.functions import create_beta_list, create_discount_list
from srl.rl.tf.model import KerasModelAddedSummary
from srl.utils.common import compare_less_version

from .agent57 import CommonInterfaceParameter, Config, Memory

kl = keras.layers
v216_older = compare_less_version(tf.__version__, "2.16.0")


class QNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config):
        super().__init__()
        self.input_ext_reward = config.input_ext_reward
        self.input_int_reward = config.input_int_reward
        self.input_action = config.input_action
        if not config.enable_intrinsic_reward:
            self.input_int_reward = False

        # --- input
        if config.observation_space.is_image():
            self.in_block = config.input_image_block.create_tf_block(config.observation_space, rnn=True)
        else:
            self.in_block = config.input_value_block.create_tf_block(config.observation_space, rnn=True)
        self.concat_layer = kl.Concatenate(axis=-1)

        # --- lstm
        self.lstm_layer = kl.LSTM(
            config.lstm_units,
            return_sequences=True,
            return_state=True,
        )

        # --- out
        self.hidden_block = config.hidden_block.create_tf_block(config.action_space.n, rnn=True)

        # build
        np_dtype = config.get_dtype("np")
        self(
            [
                self.in_block.create_dummy_data(np_dtype, timesteps=config.sequence_length),
                np.zeros((1, config.sequence_length, 1), np_dtype),
                np.zeros((1, config.sequence_length, 1), np_dtype),
                np.zeros((1, config.sequence_length, config.action_space.n), np_dtype),
                np.zeros((1, config.sequence_length, config.actor_num), np_dtype),
            ]
        )

    @tf.function()
    def call(self, x, hidden_states=None, training=False):
        return self._call(x, hidden_states, training=training)

    def _call(self, inputs, hidden_states=None, training=False):
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

        # lstm
        x, h, c = self.lstm_layer(x, initial_state=hidden_states, training=training)

        # out
        x = self.hidden_block(x, training=training)
        return x, [h, c]

    def get_initial_state(self, batch_size=1):
        if v216_older:
            return self.lstm_layer.cell.get_initial_state(batch_size=batch_size, dtype=self.dtype)
        else:
            return self.lstm_layer.cell.get_initial_state(batch_size)


# ------------------------------------------------------
# エピソード記憶部(episodic_reward)
# ------------------------------------------------------
class EmbeddingNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config):
        super().__init__()

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
        np_dtype = config.get_dtype("np")
        self(
            [
                self.in_block.create_dummy_data(np_dtype),
                np.zeros((1,) + config.observation_space.shape, np_dtype),
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
    def __init__(self, config: Config):
        super().__init__()

        if config.observation_space.is_value():
            self.in_block = config.input_value_block.create_tf_block(config.observation_space)
        elif config.observation_space.is_image():
            self.in_block = config.input_image_block.create_tf_block(config.observation_space)
        else:
            raise ValueError(config.observation_space)

        self.hidden_block = config.lifelong_hidden_block.create_tf_block()
        self.hidden_normalize = kl.LayerNormalization()

        # build
        np_dtype = config.get_dtype("np")
        self(np.zeros((1,) + config.observation_space.shape, np_dtype))
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


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self):
        self.q_ext_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr_ext_scheduler.apply_tf_scheduler(self.config.lr_ext))
        self.q_int_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr_int_scheduler.apply_tf_scheduler(self.config.lr_int))
        self.q_loss = keras.losses.Huber()

        self.emb_optimizer = keras.optimizers.Adam(learning_rate=self.config.episodic_lr_scheduler.apply_tf_scheduler(self.config.episodic_lr))
        self.lifelong_optimizer = keras.optimizers.Adam(learning_rate=self.config.lifelong_lr_scheduler.apply_tf_scheduler(self.config.lifelong_lr))

        self.beta_list = create_beta_list(self.config.actor_num)
        self.discount_list = create_discount_list(self.config.actor_num)

        self.sync_count = 0

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return
        batches, weights, update_args = batches

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
        ) = self.parameter.change_batches_format(batches, weights)

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
        td_error_ext, ext_loss = self._train_q(
            self.parameter.q_ext_online,
            self.parameter.q_ext_target,
            self.q_ext_optimizer,
            step_rewards_ext,
            hidden_states_ext,
            hidden_states_ext_t,
            *_params,
        )
        self.info["ext_loss"] = ext_loss.numpy()

        if self.config.enable_intrinsic_reward:
            td_error_int, int_loss = self._train_q(
                self.parameter.q_int_online,
                self.parameter.q_int_target,
                self.q_int_optimizer,
                step_rewards_int,
                hidden_states_int,
                hidden_states_int_t,
                *_params,
            )
            self.info["int_loss"] = int_loss.numpy()

            # embedding lifelong (batch, step, x) -> (batch, x)
            one_states = instep_states[:, 0, ...]
            one_n_states = instep_states[:, 1, ...]
            one_actions_onehot = instep_actions_onehot[:, 0, :].astype(np.float32)

            # ----------------------------------------
            # embedding network
            # ----------------------------------------
            with tf.GradientTape() as tape:
                emb_loss = self.parameter.emb_network.compute_train_loss(one_states, one_n_states, one_actions_onehot)
            grads = tape.gradient(emb_loss, self.parameter.emb_network.trainable_variables)
            self.emb_optimizer.apply_gradients(zip(grads, self.parameter.emb_network.trainable_variables))
            self.info["emb_loss"] = emb_loss.numpy()

            # ----------------------------------------
            # lifelong network
            # ----------------------------------------
            lifelong_target_val = self.parameter.lifelong_target(one_states)
            with tf.GradientTape() as tape:
                lifelong_loss = self.parameter.lifelong_train.compute_train_loss(one_states, lifelong_target_val)
            grads = tape.gradient(lifelong_loss, self.parameter.lifelong_train.trainable_variables)
            self.lifelong_optimizer.apply_gradients(zip(grads, self.parameter.lifelong_train.trainable_variables))
            self.info["lifelong_loss"] = lifelong_loss.numpy()

        else:
            td_error_int = 0

        if self.config.disable_int_priority:
            priorities = np.abs(td_error_ext)
        else:
            priorities = np.abs(td_error_ext + beta_list * td_error_int)

        self.memory.update(update_args, priorities, self.train_count)

        # --- sync target
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_ext_target.set_weights(self.parameter.q_ext_online.get_weights())
            self.parameter.q_int_target.set_weights(self.parameter.q_int_online.get_weights())
            self.sync_count += 1

        self.info["sync"] = self.sync_count
        self.train_count += 1

    def _train_q(
        self,
        model_q_online,
        model_q_target,
        optimizer,
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

        td_errors = np.mean(action_q - target_q, axis=0)
        return td_errors, loss
