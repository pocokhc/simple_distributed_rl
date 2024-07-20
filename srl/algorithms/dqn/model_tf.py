from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.trainer import RLTrainer
from srl.rl.schedulers.scheduler import SchedulerConfig
from srl.rl.tf.model import KerasModelAddedSummary

from .dqn import CommonInterfaceParameter, Config, Memory

kl = keras.layers


# ------------------------------------------------------
# network
# ------------------------------------------------------
class QNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        self.in_block = config.create_input_block_tf()
        self.hidden_block = config.hidden_block.create_block_tf()
        self.out_layer = kl.Dense(config.action_space.n, kernel_initializer="truncated_normal")

        # build
        self(np.zeros(self.in_block.create_batch_shape((1,)), config.dtype))

        self.loss_func = keras.losses.Huber()

    def call(self, x, training=False):
        x = self.in_block(x, training=training)
        x = self.hidden_block(x, training=training)
        x = self.out_layer(x)
        return x

    @tf.function
    def compute_train_loss(self, state, onehot_action, target_q):
        q = self(state, training=True)
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

    # -------------------------------------
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

        self.lr_sch = SchedulerConfig.create_scheduler(self.config.lr)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_sch.get_rate())

        self.sync_count = 0

    def implement_thread_train(self) -> bool:
        return True

    def thread_train_setup(self):
        if self.memory.is_warmup_needed():
            return None
        batchs = self.memory.sample()
        state, n_state, onehot_action, reward, undone, next_invalid_actions = zip(*batchs)
        state = self.parameter.q_online.in_block.create_batch_stack_data(state)
        onehot_action = np.asarray(onehot_action, dtype=self.config.dtype)
        reward = np.array(reward, dtype=self.config.dtype)
        undone = np.array(undone)

        target_q = self.parameter.calc_target_q(
            len(batchs),
            n_state,
            reward,
            undone,
            next_invalid_actions,
        )
        return state, onehot_action, target_q

    def thread_train(self, setup_data):
        state, onehot_action, target_q = setup_data

        with tf.GradientTape() as tape:
            loss = self.parameter.q_online.compute_train_loss(state, onehot_action, target_q)
        grad = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.parameter.q_online.trainable_variables))
        self.info["loss"] = loss.numpy()

        # --- targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())
            self.sync_count += 1

        self.info["sync"] = self.sync_count
        self.train_count += 1

        return None

    def thread_train_teardown(self, train_data):
        if self.lr_sch.update(self.train_count):
            self.optimizer.learning_rate = self.lr_sch.get_rate()
