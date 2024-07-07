from tensorflow import keras

from srl.rl.tf.model import KerasModelAddedSummary

kl = keras.layers

"""
R2D3 Image Model
https://arxiv.org/abs/1909.01387
"""


class R2D3ImageBlock(KerasModelAddedSummary):
    def __init__(
        self,
        filters: int = 16,
        activation: str = "relu",
        enable_rnn: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # enable_rnnはmax_pooling2dで必要

        self.res1 = ResBlock(filters, activation, enable_rnn)
        self.res2 = ResBlock(filters * 2, activation, enable_rnn)
        self.res3 = ResBlock(filters * 2, activation, enable_rnn)
        self.act = kl.Activation(activation)

    def call(self, x, training=False):
        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        x = self.res3(x, training=training)
        x = self.act(x)
        return x


class ResBlock(KerasModelAddedSummary):
    def __init__(
        self,
        filters,
        activation,
        enable_rnn,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.conv = kl.Conv2D(filters, (3, 3), strides=(1, 1), padding="same")
        self.pool = kl.MaxPooling2D((3, 3), strides=(2, 2), padding="same")
        self.res1 = ResidualBlock(filters, activation, enable_rnn)
        self.res2 = ResidualBlock(filters, activation, enable_rnn)

        if enable_rnn:
            self.conv = kl.TimeDistributed(self.conv)
            self.pool = kl.TimeDistributed(self.pool)

    def call(self, x, training=False):
        x = self.conv(x, training=training)
        x = self.pool(x, training=training)
        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        return x


class ResidualBlock(KerasModelAddedSummary):
    def __init__(
        self,
        n_filter,
        activation,
        enable_rnn,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.act1 = kl.Activation(activation)
        self.conv1 = kl.Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same")
        self.act2 = kl.Activation(activation)
        self.conv2 = kl.Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same")
        self.add = kl.Add()

        if enable_rnn:
            self.conv1 = kl.TimeDistributed(self.conv1)
            self.conv2 = kl.TimeDistributed(self.conv2)

    def call(self, x, training=False):
        x1 = self.act1(x)
        x1 = self.conv1(x1, training=training)
        x1 = self.act2(x1)
        x1 = self.conv2(x1, training=training)
        return self.add([x, x1])


if __name__ == "__main__":
    m = R2D3ImageBlock(enable_time_distributed_layer=True)
    m.build((None, None, 96, 72, 3))
    m.summary(expand_nested=True)
