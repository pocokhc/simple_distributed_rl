from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.trainer import RLTrainer
from srl.base.spaces.multi import MultiSpace
from srl.rl.tf.blocks.input_multi_block import InputMultiBlockConcat
from srl.rl.tf.model import KerasModelAddedSummary

from .dqn import CommonInterfaceParameter, Config, Memory

kl = keras.layers


# ------------------------------------------------------
# network
# ------------------------------------------------------
class QNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        if config.observation_space.is_value():
            self.in_block = config.input_value_block.create_tf_block(config.observation_space)
        elif config.observation_space.is_image():
            self.in_block = config.input_image_block.create_tf_block(config.observation_space)
        elif config.observation_space.is_multi():
            space = config.observation_space
            assert isinstance(space, MultiSpace)
            self.in_block = InputMultiBlockConcat(
                space,
                config.input_value_block,
                config.input_image_block,
                reshape_for_rnn=[False] * len(space.spaces),
            )
        else:
            raise ValueError(config.observation_space)

        self.hidden_block = config.hidden_block.create_tf_block()
        self.out_layer = kl.Dense(config.action_space.n, kernel_initializer="truncated_normal")

        # build
        self(self.in_block.create_dummy_data(config.get_dtype("np")))

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

    # -------------------------------------
    def pred_single_q(self, state) -> np.ndarray:
        return self.q_online(self.q_online.in_block.to_tf_one_batch(state, self.tf_dtype)).numpy()[0]

    def pred_batch_q(self, state) -> np.ndarray:
        return self.q_online(self.q_online.in_block.to_tf_batches(state, self.tf_dtype)).numpy()

    def pred_batch_target_q(self, state) -> np.ndarray:
        return self.q_target(self.q_online.in_block.to_tf_batches(state, self.tf_dtype)).numpy()


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self):
        lr = self.config.lr_scheduler.apply_tf_scheduler(self.config.lr)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

        self.tf_dtype = self.config.get_dtype("tf")
        self.np_dtype = self.config.get_dtype("np")
        self.sync_count = 0

    def train(self):
        batches = self.memory.sample()
        if batches is None:
            return
        (
            state,
            n_state,
            onehot_action,
            reward,
            undone,
            next_invalid_actions,
        ) = zip(*batches)

        state = self.parameter.q_online.in_block.to_tf_batches(state, self.tf_dtype)
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
