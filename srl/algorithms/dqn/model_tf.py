from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.base import RLTrainer
from srl.rl.models.tf import helper
from srl.rl.models.tf.input_block import InputBlock
from srl.rl.models.tf.model import KerasModelAddedSummary

from .dqn import CommonInterfaceParameter, Config

kl = keras.layers


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        # input
        self.input_block = InputBlock(config.input_value_block, config.input_image_block, config.observation_space)

        # hidden
        self.hidden_block = config.hidden_block.create_block_tf()

        # out layer
        self.out_layer = kl.Dense(
            config.action_num,
            kernel_initializer="truncated_normal",
            bias_initializer="truncated_normal",
        )

        self.loss_func = keras.losses.Huber()

        # build
        self.build(self.create_batch_shape(config.observation_shape, (None,)))

    def call(self, x, training=False):
        x = self.input_block(x, training)
        x = self.hidden_block(x, training)
        return self.out_layer(x)

    @tf.function
    def compute_train_loss(self, state, onehot_action, target_q):
        q = self(state)
        q = tf.reduce_sum(q * onehot_action, axis=1)
        loss = self.loss_func(target_q, q)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss


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
    def create_batch_data(self, state):
        return helper.create_batch_data(state, self.config.observation_space)

    def predict_q(self, state) -> np.ndarray:
        return self.q_online(state).numpy()

    def predict_target_q(self, state) -> np.ndarray:
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
        batchs = self.memory.sample(self.batch_size)
        state, n_state, onehot_action, reward, done, next_invalid_actions = zip(*batchs)
        state = helper.stack_batch_data(state, self.config.observation_space)
        n_state = helper.stack_batch_data(n_state, self.config.observation_space)
        onehot_action = np.asarray(onehot_action, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done)

        target_q = self.parameter.calc_target_q(
            len(batchs),
            n_state,
            reward,
            done,
            next_invalid_actions,
        )

        with tf.GradientTape() as tape:
            loss = self.parameter.q_online.compute_train_loss(state, onehot_action, target_q)
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
