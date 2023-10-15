from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.base import RLTrainer
from srl.rl.models.tf.input_block import InputBlock

from .dqn import CommonInterfaceParameter, Config

kl = keras.layers


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(keras.Model):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.__input_shape = config.observation_shape

        # input
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)

        # image
        if self.in_block.use_image_layer:
            self.image_block = config.image_block.create_block_tf(enable_time_distributed_layer=False)
            self.image_flatten = kl.Flatten()

        # hidden
        self.hidden_block = config.hidden_block.create_block_tf()

        # out layer
        self.out_layer = kl.Dense(
            config.action_num,
            activation="linear",
            kernel_initializer="truncated_normal",
            bias_initializer="truncated_normal",
        )

        # build
        self.build((None,) + config.observation_shape)

    @tf.function
    def call(self, x, training=False):
        return self._call(x, training)

    def _call(self, x, training=False):
        x = self.in_block(x, training=training)
        if self.in_block.use_image_layer:
            x = self.image_block(x, training=training)
            x = self.image_flatten(x)
        x = self.hidden_block(x, training=training)
        x = self.out_layer(x, training=training)
        return x

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def summary(self, name="", **kwargs):
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

        self.q_online = _QNetwork(self.config, name="Q_online")
        self.q_target = _QNetwork(self.config, name="Q_target")
        self.q_target.set_weights(self.q_online.get_weights())

    def call_restore(self, data: Any, **kwargs) -> None:
        self.q_online.set_weights(data)
        self.q_target.set_weights(data)

    def call_backup(self, **kwargs) -> Any:
        return self.q_online.get_weights()

    def summary(self, **kwargs):
        self.q_online.summary(**kwargs)

    # -------------------------------------

    def predict_q(self, state: np.ndarray) -> np.ndarray:
        return self.q_online.call(state).numpy()  # type:ignore , ignore check "None"

    def predict_target_q(self, state: np.ndarray) -> np.ndarray:
        return self.q_target(state).numpy()  # type:ignore , ignore check "None"


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_sch = self.config.lr.create_schedulers()

        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch.get_rate(0))
        self.loss_func = keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)

        self.sync_count = 0

    def train_on_batchs(self, memory_sample_return) -> None:
        indices, batchs, weights = memory_sample_return

        target_q, states, onehot_actions = self.parameter.calc_target_q(batchs, training=True)

        with tf.GradientTape() as tape:
            q = self.parameter.q_online(states)
            q = tf.reduce_sum(q * onehot_actions, axis=1)

            loss = self.loss_func(target_q * weights, q * weights)
            loss += tf.nn.scale_regularization_loss(self.parameter.q_online.losses)

        grads = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.q_online.trainable_variables))

        lr = self.lr_sch.get_rate(self.train_count)
        self.optimizer.learning_rate = lr

        # --- update
        priorities = np.abs(target_q - q)
        self.memory_update((indices, batchs, priorities))

        # --- targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())
            self.sync_count += 1

        self.train_info = {
            "loss": loss.numpy(),
            "sync": self.sync_count,
            "lr": lr,
        }
        self.train_count += 1
