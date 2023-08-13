import numpy as np
import tensorflow as tf


class NoisyDense(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        sigma: float = 0.5,
        activation=None,
        kernel_initializer="random_normal",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.sigma = sigma
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        p = input_shape[-1]
        self.w_mu = self.add_weight(
            name="w_mu",
            shape=(p, self.units),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.w_sigma = self.add_weight(
            name="w_sigma",
            shape=(p, self.units),
            initializer=tf.keras.initializers.Constant(self.sigma / np.sqrt(p)),
            trainable=True,
        )

        self.b_mu = self.add_weight(
            name="b_mu",
            shape=(1, self.units),
            initializer=self.bias_initializer,
            trainable=True,
        )
        self.b_sigma = self.add_weight(
            name="b_sigma",
            shape=(1, self.units),
            initializer=tf.keras.initializers.Constant(self.sigma / np.sqrt(p)),
            trainable=True,
        )

        self.built = True
        super().build(input_shape)

    def call(self, inputs):
        w_noise = tf.random.normal(self.w_mu.shape)
        b_noise = tf.random.normal(self.b_mu.shape)

        w = self.w_mu + self.w_sigma * w_noise
        b = self.b_mu + self.b_sigma * b_noise

        output = tf.matmul(inputs, w) + b
        if self.activation is not None:
            output = self.activation(output)
        return output
