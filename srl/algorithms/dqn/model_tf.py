from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.trainer import RLTrainer
from srl.rl.tf.model import KerasModelAddedSummary

from .dqn import CommonInterfaceParameter, Config, Memory

kl = keras.layers


# ------------------------------------------------------
# network
# ------------------------------------------------------
class QNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        self.in_block = config.input_block.create_tf_block(config)
        self.hidden_block = config.hidden_block.create_tf_block()
        self.out_layer = kl.Dense(config.action_space.n, kernel_initializer="truncated_normal")

        # build
        self(config.input_block.create_tf_dummy_data(config))

        self.loss_func = keras.losses.Huber()

    def call(self, x, training=False):
        x = self.in_block(x, training=training)
        x = self.hidden_block(x, training=training)
        x = self.out_layer(x)
        return x

    @tf.function
    def compute_train_loss(self, state, onehot_action, target_q, weights):
        q = self(state, training=True)
        q = tf.reduce_sum(q * onehot_action, axis=1)
        loss = self.loss_func(target_q * weights, q * weights)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss, q


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(CommonInterfaceParameter):
    def setup(self):
        super().setup()
        self.q_online = QNetwork(self.config, name="Q_online")
        self.q_target = QNetwork(self.config, name="Q_target")
        self.q_target.set_weights(self.q_online.get_weights())
        self.np_dtype = self.config.get_dtype("np")

    def call_restore(self, data: Any, **kwargs) -> None:
        self.q_online.set_weights(data)
        self.q_target.set_weights(data)

    def call_backup(self, **kwargs) -> Any:
        return self.q_online.get_weights()

    def summary(self, **kwargs):
        self.q_online.summary(**kwargs)

    # -------------------------------------
    def pred_q(self, state: np.ndarray) -> np.ndarray:
        state = tf.convert_to_tensor(np.asarray(state, dtype=self.np_dtype))
        return self.q_online(state).numpy()

    def pred_target_q(self, state: np.ndarray) -> np.ndarray:
        state = tf.convert_to_tensor(np.asarray(state, dtype=self.np_dtype))
        return self.q_target(state).numpy()


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self):
        lr = self.config.lr_scheduler.apply_tf_scheduler(self.config.lr)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.np_dtype = self.config.get_dtype("np")
        self.sync_count = 0

    def train(self):
        batches = self.memory.sample()
        if batches is None:
            return
        batches, weights, update_args = batches

        (
            state,
            n_state,
            onehot_action,
            reward,
            undone,
            next_invalid_actions,
        ) = zip(*batches)

        state = tf.convert_to_tensor(np.asarray(state, dtype=self.np_dtype))
        n_state = np.asarray(n_state, dtype=self.np_dtype)
        onehot_action = np.asarray(onehot_action, dtype=self.np_dtype)
        reward = np.array(reward, dtype=self.np_dtype)
        undone = np.array(undone)

        target_q = self.parameter.calc_target_q(
            len(batches),
            n_state,
            reward,
            undone,
            next_invalid_actions,
        )

        with tf.GradientTape() as tape:
            loss, q = self.parameter.q_online.compute_train_loss(state, onehot_action, target_q, weights)
        grad = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.parameter.q_online.trainable_variables))
        self.info["loss"] = loss.numpy()

        # --- update
        priorities = np.abs(target_q - q)
        self.memory.update(update_args, priorities, self.train_count)

        # --- targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())
            self.sync_count += 1

        self.info["sync"] = self.sync_count
        self.train_count += 1
