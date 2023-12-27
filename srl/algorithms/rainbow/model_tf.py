from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import RLTypes
from srl.base.rl.base import RLTrainer
from srl.rl.models.tf.input_block import InputImageBlock

from .rainbow import CommonInterfaceParameter, Config
from .rainbow_nomultisteps import calc_target_q

kl = keras.layers


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(keras.Model):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        # in block
        self.in_img_block = None
        if config.observation_type == RLTypes.IMAGE:
            self.in_img_block = InputImageBlock(config.observation_shape, config.env_observation_type)
            self.img_block = config.image_block.create_block_tf()
        self.flat_layer = kl.Flatten()

        # out
        self.dueling_block = config.dueling_network.create_block_tf(
            config.action_num,
            enable_noisy_dense=config.enable_noisy_dense,
        )

        # build
        self._in_shape = config.observation_shape
        self.build((None,) + self._in_shape)

        self.loss_func = keras.losses.Huber()

    def call(self, x, training=False):
        if self.in_img_block is not None:
            x = self.in_img_block(x, training)
            x = self.img_block(x, training)
        x = self.flat_layer(x)
        x = self.dueling_block(x, training=training)
        return x

    @tf.function
    def compute_train_loss(self, state, onehot_action, target_q, weights):
        q = self(state, training=True)
        q = tf.reduce_sum(q * onehot_action, axis=1)
        loss = self.loss_func(target_q * weights, q * weights)
        loss += tf.reduce_sum(self.losses)
        return loss, q

    def summary(self, name="", **kwargs):
        if self.in_img_block is not None:
            self.in_img_block.init_model_graph()
            self.img_block.init_model_graph()
        self.dueling_block.init_model_graph()

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

    # ----------------------------------------------
    def predict_q(self, state: np.ndarray) -> np.ndarray:
        return self.q_online(state).numpy()  # type:ignore , "numpy" is not a known member of "None"

    def predict_target_q(self, state: np.ndarray) -> np.ndarray:
        return self.q_target(state).numpy()  # type:ignore , "numpy" is not a known member of "None"


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
        self.sync_count = 0

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        indices, batchs, weights = self.memory.sample(self.batch_size, self.train_count)

        if self.config.multisteps == 1:
            target_q, states, onehot_actions = calc_target_q(self.parameter, batchs, training=True)
        else:
            target_q, states, onehot_actions = self.parameter.calc_target_q(batchs, training=True)

        with tf.GradientTape() as tape:
            loss, q = self.parameter.q_online.compute_train_loss(states, onehot_actions, target_q, weights)
        grads = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.q_online.trainable_variables))

        if self.lr_sch.update(self.train_count):
            self.optimizer.learning_rate = self.lr_sch.get_rate()

        # --- update
        priorities = np.abs(target_q - q)
        self.memory.update((indices, batchs, priorities))

        # --- sync target
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())
            self.sync_count += 1

        self.train_count += 1
        self.train_info = {
            "loss": loss.numpy(),
            "sync": self.sync_count,
            "lr": self.lr_sch.get_rate(),
        }
