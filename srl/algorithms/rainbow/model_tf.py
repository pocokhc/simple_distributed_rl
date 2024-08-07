from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.trainer import RLTrainer
from srl.rl.schedulers.scheduler import SchedulerConfig
from srl.rl.tf.model import KerasModelAddedSummary

from .rainbow import CommonInterfaceParameter, Config, Memory
from .rainbow_nomultisteps import calc_target_q

kl = keras.layers


# ------------------------------------------------------
# network
# ------------------------------------------------------
class QNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        self.in_block = config.create_input_block_tf()
        self.hidden_block = config.hidden_block.create_block_tf(config.action_space.n)

        # build
        self(np.zeros(self.in_block.create_batch_shape((1,)), config.dtype))

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


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(CommonInterfaceParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        self.q_online = QNetwork(self.config, name="Q_online")
        self.q_target = QNetwork(self.config, name="Q_target")
        self.q_target.set_weights(self.q_online.get_weights())

    def call_restore(self, data: Any, **kwargs) -> None:
        self.q_online.set_weights(data)
        self.q_target.set_weights(data)

    def call_backup(self, **kwargs) -> Any:
        return self.q_online.get_weights()

    def summary(self, **kwargs):
        self.q_online.summary(**kwargs)

    # ----------------------------------------------
    def pred_single_q(self, state) -> np.ndarray:
        state = self.q_online.in_block.create_batch_single_data(state)
        return self.q_online(state).numpy()[0]

    def pred_batch_q(self, state) -> np.ndarray:
        state = self.q_online.in_block.create_batch_stack_data(state)
        return self.q_online(state).numpy()

    def pred_batch_target_q(self, state) -> np.ndarray:
        state = self.q_online.in_block.create_batch_stack_data(state)
        return self.q_target(state).numpy()


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer[Config, Parameter, Memory]):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_sch = SchedulerConfig.create_scheduler(self.config.lr)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch.get_rate())
        self.sync_count = 0

    def implement_thread_train(self) -> bool:
        return True

    def thread_train_setup(self):
        if self.memory.is_warmup_needed():
            return None
        batchs, weights, update_args = self.memory.sample(self.train_count)

        if self.config.multisteps == 1:
            target_q, states, onehot_actions = calc_target_q(self.parameter, batchs, training=True)
        else:
            target_q, states, onehot_actions = self.parameter.calc_target_q(batchs, training=True)

        return states, onehot_actions, target_q, weights, update_args

    def thread_train(self, setup_data):
        states, onehot_actions, target_q, weights, update_args = setup_data

        with tf.GradientTape() as tape:
            loss, q = self.parameter.q_online.compute_train_loss(states, onehot_actions, target_q, weights)
        grad = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.parameter.q_online.trainable_variables))

        # --- sync target
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())
            self.sync_count += 1
        self.info["sync"] = self.sync_count
        self.train_count += 1

        return loss.numpy(), target_q, q, update_args

    def thread_train_teardown(self, train_data):
        loss, target_q, q, update_args = train_data

        self.info["loss"] = loss

        if self.lr_sch.update(self.train_count):
            self.optimizer.learning_rate = self.lr_sch.get_rate()

        # --- update
        priorities = np.abs(target_q - q)
        self.memory.update(update_args, priorities)
