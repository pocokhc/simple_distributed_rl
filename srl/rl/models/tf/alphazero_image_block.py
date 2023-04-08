import tensorflow.keras as keras
from tensorflow.keras import layers as kl
from tensorflow.keras import regularizers

"""
Paper:
https://www.deepmind.com/blog/alphazero-shedding-new-light-on-chess-shogi-and-go
https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf

Code ref:
https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning
"""


class AlphaZeroImageBlock(keras.Model):
    def __init__(
        self,
        n_blocks: int = 19,
        filters: int = 256,
        kernel_size=(3, 3),
        l2: float = 0.0001,
        use_layer_normalization: bool = False,
    ):
        super().__init__()

        self.conv1 = kl.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(l2),
        )
        if use_layer_normalization:
            self.bn1 = kl.LayerNormalization()
        else:
            self.bn1 = kl.BatchNormalization()
        self.act1 = kl.ReLU()

        self.resblocks = [_ResidualBlock(filters, kernel_size, l2, use_layer_normalization) for _ in range(n_blocks)]

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        for resblock in self.resblocks:
            x = resblock(x)
        return x

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def init_model_graph(self, name: str = ""):
        [r.init_model_graph() for r in self.resblocks]

        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        keras.Model(inputs=x, outputs=self.call(x), name=name)


class _ResidualBlock(keras.Model):
    def __init__(
        self,
        filters: int,
        kernel_size=(3, 3),
        l2: float = 0.0001,
        use_layer_normalization: bool = False,
    ):
        super().__init__()

        self.conv1 = kl.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(l2),
        )
        if use_layer_normalization:
            self.bn1 = kl.LayerNormalization()
        else:
            self.bn1 = kl.BatchNormalization()
        self.act1 = kl.ReLU()
        self.conv2 = kl.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(l2),
        )
        if use_layer_normalization:
            self.bn2 = kl.LayerNormalization()
        else:
            self.bn2 = kl.BatchNormalization()
        self.act2 = kl.ReLU()

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x = x + x1
        x = self.act2(x)
        return x

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def init_model_graph(self, name: str = ""):
        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        keras.Model(inputs=x, outputs=self.call(x), name=name)


if __name__ == "__main__":
    m = AlphaZeroImageBlock(n_blocks=1)
    m.build((None, 96, 72, 3))
    m.init_model_graph()
    m.summary(expand_nested=True)
