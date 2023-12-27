from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import RLTypes
from srl.base.rl.base import RLTrainer
from srl.rl.models.tf.input_block import InputImageBlock

from .dqn import CommonInterfaceParameter, Config

kl = keras.layers


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(keras.Model):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        # input
        self.in_img_block = None
        if config.observation_type == RLTypes.IMAGE:
            self.in_img_block = InputImageBlock(config.observation_shape, config.env_observation_type)
            self.img_block = config.image_block.create_block_tf()
        self.flat_layer = kl.Flatten()

        # hidden
        self.h_block = config.hidden_block.create_block_tf()

        # out layer
        self.out_layer = kl.Dense(
            config.action_num,
            kernel_initializer="truncated_normal",
            bias_initializer="truncated_normal",
        )

        self.loss_func = keras.losses.Huber()

        # build
        self._in_shape = config.observation_shape
        self.build((None,) + self._in_shape)

    def call(self, x, training=False):
        if self.in_img_block is not None:
            x = self.in_img_block(x, training)
            x = self.img_block(x, training)
        x = self.flat_layer(x)
        x = self.h_block(x, training)
        return self.out_layer(x)

    @tf.function
    def compute_train_loss(self, state, onehot_action, target_q):
        q = self(state)
        q = tf.reduce_sum(q * onehot_action, axis=1)
        loss = self.loss_func(target_q, q)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss

    def summary(self, name="", **kwargs):
        if self.in_img_block is not None:
            self.in_img_block.init_model_graph()
            self.img_block.init_model_graph()
        self.h_block.init_model_graph()

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
        return self.q_online(state).numpy()

    def predict_target_q(self, state: np.ndarray) -> np.ndarray:
        return self.q_target(state).numpy()


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_sch = self.config.lr.create_schedulers()
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch.get_rate())
        self.loss_func = keras.losses.Huber()

        self.sync_count = 0

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        batchs = self.memory.sample(self.batch_size, self.train_count)

        target_q, states, onehot_actions = self.parameter.calc_target_q(batchs, training=True)
        with tf.GradientTape() as tape:
            loss = self.parameter.q_online.compute_train_loss(states, onehot_actions, target_q)
        grads = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.q_online.trainable_variables))

        if self.lr_sch.update(self.train_count):
            self.optimizer.learning_rate = self.lr_sch.get_rate()

        # --- targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())
            self.sync_count += 1

        self.train_info = {
            "loss": loss.numpy(),
            "sync": self.sync_count,
            "lr": self.lr_sch.get_rate(),
        }
        self.train_count += 1
