from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.trainer import RLTrainer
from srl.rl.tf.model import KerasModelAddedSummary

from .rainbow import CommonInterfaceParameter, Config, Memory
from .rainbow_nomultisteps import calc_target_q

kl = keras.layers


class QNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        if config.observation_space.is_value():
            self.in_block = config.input_value_block.create_tf_block(config.observation_space)
        elif config.observation_space.is_image():
            self.in_block = config.input_image_block.create_tf_block(config.observation_space)
        else:
            raise ValueError(config.observation_space)

        self.hidden_block = config.hidden_block.create_tf_block(config.action_space.n)

        # build
        self(self.in_block.create_dummy_data(config.get_dtype("np")))

        self.loss_func = keras.losses.Huber()

    def call(self, x, training=False):
        x = self.in_block(x, training=training)
        x = self.hidden_block(x, training=training)
        return x

    @tf.function
    def compute_train_loss(self, state, onehot_action, target_q, weights):
        q = self(state, training=True)
        q = tf.reduce_sum(q * onehot_action, axis=1)
        loss = self.loss_func(target_q * weights, q * weights)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss, q


class Parameter(CommonInterfaceParameter):
    def setup(self):
        super().setup()
        self.q_online = QNetwork(self.config, name="Q_online")
        self.q_target = QNetwork(self.config, name="Q_target")
        self.q_target.set_weights(self.q_online.get_weights())
        self.tf_dtype = self.config.get_dtype("tf")

    def call_restore(self, data: Any, **kwargs) -> None:
        self.q_online.set_weights(data)
        self.q_target.set_weights(data)

    def call_backup(self, **kwargs) -> Any:
        return self.q_online.get_weights()

    def summary(self, **kwargs):
        self.q_online.summary(**kwargs)

    # ----------------------------------------------
    def pred_single_q(self, state) -> np.ndarray:
        return self.q_online(self.q_online.in_block.to_tf_one_batch(state, self.tf_dtype)).numpy()[0]

    def pred_batch_q(self, state) -> np.ndarray:
        return self.q_online(self.q_online.in_block.to_tf_batches(state, self.tf_dtype)).numpy()

    def pred_batch_target_q(self, state) -> np.ndarray:
        return self.q_target(self.q_online.in_block.to_tf_batches(state, self.tf_dtype)).numpy()


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        lr = self.config.lr_scheduler.apply_tf_scheduler(self.config.lr)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.sync_count = 0

    def train(self):
        batches = self.memory.sample()
        if batches is None:
            return
        batches, weights, update_args = batches

        if self.config.multisteps == 1:
            target_q, states, onehot_actions = calc_target_q(self.parameter, batches, training=True)
        else:
            target_q, states, onehot_actions = self.parameter.calc_target_q(batches, training=True)

        with tf.GradientTape() as tape:
            loss, q = self.parameter.q_online.compute_train_loss(states, onehot_actions, target_q, weights)
        grad = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.parameter.q_online.trainable_variables))

        # --- sync target
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())
            self.sync_count += 1
        self.info["loss"] = loss.numpy()
        self.info["sync"] = self.sync_count
        self.train_count += 1

        # --- update
        priorities = np.abs(target_q - q)
        self.memory.update(update_args, priorities, self.train_count)
