import tensorflow.keras as keras
from tensorflow.keras import layers as kl
from tensorflow.keras import regularizers

"""
Paper
https://arxiv.org/abs/1911.08265

Ref
https://github.com/horoiwa/deep_reinforcement_learning_gallery
"""


class MuZeroAtariBlock(keras.Model):
    def __init__(
        self,
        base_filters: int = 128,
        kernel_size=(3, 3),
        l2: float = 0.0001,
    ):
        super().__init__()

        self.conv1 = kl.Conv2D(
            base_filters,
            kernel_size=kernel_size,
            strides=2,
            padding="same",
            activation="relu",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(l2),
        )
        self.resblocks1 = _ResidualBlock(base_filters, kernel_size, l2)
        self.resblocks2 = _ResidualBlock(base_filters, kernel_size, l2)
        self.conv2 = kl.Conv2D(
            base_filters * 2,
            kernel_size=kernel_size,
            strides=2,
            padding="same",
            activation="relu",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(l2),
        )
        self.resblocks3 = _ResidualBlock(base_filters * 2, kernel_size, l2)
        self.resblocks4 = _ResidualBlock(base_filters * 2, kernel_size, l2)
        self.resblocks5 = _ResidualBlock(base_filters * 2, kernel_size, l2)
        self.pool1 = kl.AveragePooling2D(pool_size=3, strides=2, padding="same")
        self.resblocks6 = _ResidualBlock(base_filters * 2, kernel_size, l2)
        self.resblocks7 = _ResidualBlock(base_filters * 2, kernel_size, l2)
        self.resblocks8 = _ResidualBlock(base_filters * 2, kernel_size, l2)
        self.pool2 = kl.AveragePooling2D(pool_size=3, strides=2, padding="same")

    def call(self, x):
        x = self.conv1(x)
        x = self.resblocks1(x)
        x = self.resblocks2(x)
        x = self.conv2(x)
        x = self.resblocks3(x)
        x = self.resblocks4(x)
        x = self.resblocks5(x)
        x = self.pool1(x)
        x = self.resblocks6(x)
        x = self.resblocks7(x)
        x = self.resblocks8(x)
        x = self.pool2(x)
        return x


class _ResidualBlock(keras.Model):
    def __init__(self, filters, kernel_size, l2):
        super().__init__()

        self.conv1 = kl.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(l2),
        )
        self.bn1 = kl.BatchNormalization()
        self.relu1 = kl.LeakyReLU()
        self.conv2 = kl.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(l2),
        )
        self.bn2 = kl.BatchNormalization()
        self.relu2 = kl.LeakyReLU()

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x = x + x1
        x = self.relu2(x)
        return x


if __name__ == "__main__":
    in_ = c = keras.Input((96, 96, 3))
    c = MuZeroAtariBlock()(c)
    model = keras.Model(in_, c)
    model.summary(expand_nested=True)
