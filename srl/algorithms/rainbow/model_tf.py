from typing import Any, cast

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.base import RLTrainer
from srl.rl.models.tf.dueling_network import DuelingNetworkBlock
from srl.rl.models.tf.input_block import InputBlock

from .rainbow import CommonInterfaceParameter, Config, RemoteMemory
from .rainbow_nomultisteps import calc_target_q

kl = keras.layers


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(keras.Model):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        # input
        self.in_block = InputBlock(config.observation_shape, config.env_observation_type)

        # image
        if self.in_block.use_image_layer:
            self.image_block = config.image_block.create_block_tf(enable_time_distributed_layer=False)
            self.image_flatten = kl.Flatten()

        if config.enable_noisy_dense:
            # TensorFlow Addons Wind Down : https://github.com/tensorflow/addons/issues/2807
            # if common.compare_less_package_version("tensorflow_addons", "2.11.0"):

            import tensorflow_addons as tfa

            _Dense = tfa.layers.NoisyDense
        else:
            _Dense = kl.Dense

        # hidden
        self.hidden_layers = []
        for i in range(len(config.hidden_layer_sizes) - 1):
            self.hidden_layers.append(
                _Dense(
                    config.hidden_layer_sizes[i],
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )

        # out
        self.enable_dueling_network = config.enable_dueling_network
        if config.enable_dueling_network:
            self.dueling_block = DuelingNetworkBlock(
                config.action_num,
                config.hidden_layer_sizes[-1],
                config.dueling_network_type,
                activation="relu",
                enable_noisy_dense=config.enable_noisy_dense,
            )
        else:
            self.out_layers = [
                _Dense(config.hidden_layer_sizes[-1], activation="relu", kernel_initializer="he_normal"),
                _Dense(config.action_num, kernel_initializer="truncated_normal", bias_initializer="truncated_normal"),
            ]

        # build
        self.build((None,) + config.observation_shape)

    def call(self, x, training=False):
        x = self.in_block(x, training=training)
        if self.in_block.use_image_layer:
            x = self.image_block(x, training=training)
            x = self.image_flatten(x)
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        if self.enable_dueling_network:
            x = self.dueling_block(x, training=training)
        else:
            for layer in self.out_layers:
                x = layer(x, training=training)
        return x

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def summary(self, name="", **kwargs):
        if hasattr(self.in_block, "init_model_graph"):
            self.in_block.init_model_graph()
        if self.in_block.use_image_layer and hasattr(self.image_block, "init_model_graph"):
            self.image_block.init_model_graph()
        if self.enable_dueling_network and hasattr(self.dueling_block, "init_model_graph"):
            self.dueling_block.init_model_graph()

        x = kl.Input(shape=self.__input_shape[1:])
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
        self.remote_memory: RemoteMemory = self.remote_memory

        self.optimizer = keras.optimizers.Adam(learning_rate=self.config.lr)
        self.loss_func = keras.losses.Huber()

        self.train_count = 0
        self.sync_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):
        if self.remote_memory.length() < self.config.memory_warmup_size:
            return {}

        indices, batchs, weights = self.remote_memory.sample(self.config.batch_size, self.train_count)

        if self.config.multisteps == 1:
            target_q, states, onehot_actions = calc_target_q(self.parameter, batchs, training=True)
        else:
            target_q, states, onehot_actions = self.parameter.calc_target_q(batchs, training=True)

        with tf.GradientTape() as tape:
            q = self.parameter.q_online(states, training=True)
            q = tf.reduce_sum(q * onehot_actions, axis=1)

            loss = self.loss_func(target_q * weights, q * weights)
            loss += tf.reduce_sum(self.parameter.q_online.losses)

        grads = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.q_online.trainable_variables))

        # --- update
        td_errors = target_q - cast(Any, q).numpy()
        self.remote_memory.update(indices, batchs, td_errors)

        # --- sync target
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())
            self.sync_count += 1

        self.train_count += 1
        return {"loss": loss.numpy(), "sync": self.sync_count}
