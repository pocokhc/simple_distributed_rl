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
        base_filters: int = 128,
        kernel_size=(3, 3),
        l2: float = 0.0001,
        use_layer_normalization: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.conv1 = kl.Conv2D(
            base_filters,
            kernel_size=kernel_size,
            strides=2,
            padding="same",
            activation="relu",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(l2),
        )
        self.resblock1 = _ResidualBlock(base_filters, kernel_size, l2, use_layer_normalization)
        self.resblock2 = _ResidualBlock(base_filters, kernel_size, l2, use_layer_normalization)
        self.conv2 = kl.Conv2D(
            base_filters * 2,
            kernel_size=kernel_size,
            strides=2,
            padding="same",
            activation="relu",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(l2),
        )
        self.resblock3 = _ResidualBlock(base_filters * 2, kernel_size, l2, use_layer_normalization)
        self.resblock4 = _ResidualBlock(base_filters * 2, kernel_size, l2, use_layer_normalization)
        self.resblock5 = _ResidualBlock(base_filters * 2, kernel_size, l2, use_layer_normalization)
        self.pool1 = kl.AveragePooling2D(pool_size=3, strides=2, padding="same")
        self.resblock6 = _ResidualBlock(base_filters * 2, kernel_size, l2, use_layer_normalization)
        self.resblock7 = _ResidualBlock(base_filters * 2, kernel_size, l2, use_layer_normalization)
        self.resblock8 = _ResidualBlock(base_filters * 2, kernel_size, l2, use_layer_normalization)
        self.pool2 = kl.AveragePooling2D(pool_size=3, strides=2, padding="same")

    def call(self, x):
        x = self.conv1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.conv2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.pool1(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.pool2(x)
        return x


class _ResidualBlock(keras.Model):
    def __init__(
        self,
        filters,
        kernel_size,
        l2,
        use_layer_normalization: bool = True,
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
        self.relu1 = kl.ReLU()
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
        self.relu2 = kl.ReLU()

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x = x + x1
        x = self.relu2(x)
        return x
