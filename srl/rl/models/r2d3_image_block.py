import tensorflow.keras as keras
from tensorflow.keras import layers as kl

"""
R2D3 Image Model
https://arxiv.org/abs/1909.01387
"""


class R2D3ImageBlock(kl.Layer):
    def __init__(self):
        super().__init__()

        self.res1 = _ResBlock(16)
        self.res2 = _ResBlock(32)
        self.res3 = _ResBlock(32)
        self.relu = kl.Activation("relu")
        self.flatten = kl.Flatten()
        self.dense = kl.Dense(256, activation="relu")

    def call(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class _ResBlock(kl.Layer):
    def __init__(self, n_filter):
        super().__init__()

        self.conv = kl.Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same")
        self.pool = kl.MaxPooling2D((3, 3), strides=(2, 2), padding="same")
        self.res1 = _ResidualBlock(n_filter)
        self.res2 = _ResidualBlock(n_filter)

    def call(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class _ResidualBlock(kl.Layer):
    def __init__(self, n_filter):
        super().__init__()

        self.relu1 = kl.Activation("relu")
        self.conv1 = kl.Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same")
        self.relu2 = kl.Activation("relu")
        self.conv2 = kl.Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same")
        self.add = kl.Add()

    def call(self, x):
        x1 = self.relu1(x)
        x1 = self.conv1(x1)
        x1 = self.relu2(x1)
        x1 = self.conv2(x1)
        return self.add([x, x1])


if __name__ == "__main__":
    in_ = c = kl.Input((96, 72, 3))
    c = R2D3ImageBlock()(c)
    model = keras.Model(in_, c)
    model.summary(expand_nested=True)

    # from tensorflow.keras.utils import plot_model

    # plot_model(model, "tmp/model.png", show_shapes=True, expand_nested=True)
