from tensorflow import keras

kl = keras.layers

"""
Paper
https://arxiv.org/abs/1911.08265

Ref
https://github.com/horoiwa/deep_reinforcement_learning_gallery
"""


class MuZeroAtariBlock(keras.Model):
    def __init__(
        self,
        filters: int = 128,
        kernel_size=(3, 3),
        l2: float = 0.0001,
        activation: str = "relu",
        use_layer_normalization: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.conv1 = kl.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=2,
            padding="same",
            activation=activation,
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(l2),
        )
        self.resblock1 = _ResidualBlock(filters, kernel_size, l2, activation, use_layer_normalization)
        self.resblock2 = _ResidualBlock(filters, kernel_size, l2, activation, use_layer_normalization)
        self.conv2 = kl.Conv2D(
            filters * 2,
            kernel_size=kernel_size,
            strides=2,
            padding="same",
            activation=activation,
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(l2),
        )
        self.resblock3 = _ResidualBlock(filters * 2, kernel_size, l2, activation, use_layer_normalization)
        self.resblock4 = _ResidualBlock(filters * 2, kernel_size, l2, activation, use_layer_normalization)
        self.resblock5 = _ResidualBlock(filters * 2, kernel_size, l2, activation, use_layer_normalization)
        self.pool1 = kl.AveragePooling2D(pool_size=3, strides=2, padding="same")
        self.resblock6 = _ResidualBlock(filters * 2, kernel_size, l2, activation, use_layer_normalization)
        self.resblock7 = _ResidualBlock(filters * 2, kernel_size, l2, activation, use_layer_normalization)
        self.resblock8 = _ResidualBlock(filters * 2, kernel_size, l2, activation, use_layer_normalization)
        self.pool2 = kl.AveragePooling2D(pool_size=3, strides=2, padding="same")

    def call(self, x, training=False):
        x = self.conv1(x, training=training)
        x = self.resblock1(x, training=training)
        x = self.resblock2(x, training=training)
        x = self.conv2(x, training=training)
        x = self.resblock3(x, training=training)
        x = self.resblock4(x, training=training)
        x = self.resblock5(x, training=training)
        x = self.pool1(x, training=training)
        x = self.resblock6(x, training=training)
        x = self.resblock7(x, training=training)
        x = self.resblock8(x, training=training)
        x = self.pool2(x, training=training)
        return x


class _ResidualBlock(keras.Model):
    def __init__(
        self,
        filters,
        kernel_size,
        l2,
        activation,
        use_layer_normalization: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.conv1 = kl.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(l2),
        )
        if use_layer_normalization:
            self.bn1 = kl.LayerNormalization()
        else:
            self.bn1 = kl.BatchNormalization()
        self.act1 = kl.Activation(activation)
        self.conv2 = kl.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(l2),
        )
        if use_layer_normalization:
            self.bn2 = kl.LayerNormalization()
        else:
            self.bn2 = kl.BatchNormalization()
        self.act2 = kl.Activation(activation)

    def call(self, x, training=False):
        x1 = self.conv1(x, training=training)
        x1 = self.bn1(x1, training=training)
        x1 = self.act1(x1)
        x1 = self.conv2(x1, training=training)
        x1 = self.bn2(x1, training=training)
        x = x + x1
        x = self.act2(x)
        return x
