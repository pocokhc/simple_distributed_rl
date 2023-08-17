from tensorflow import keras

kl = keras.layers

"""
Paper:
https://www.deepmind.com/blog/alphazero-shedding-new-light-on-chess-shogi-and-go
https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf

Code ref:
https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning
https://github.com/suragnair/alpha-zero-general
"""


class AlphaZeroImageBlock(keras.Model):
    def __init__(
        self,
        n_blocks: int = 19,
        filters: int = 256,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_layers = [
            kl.BatchNormalization(),
            kl.Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                strides=1,
                padding="same",
                activation=activation,
            ),
            kl.BatchNormalization(),
        ]

        self.resblocks = [_ResidualBlock(filters, activation) for _ in range(n_blocks)]

    def call(self, x):
        for layer in self.input_layers:
            x = layer(x)
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
        activation,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.res_layers = [
            kl.Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                strides=1,
                padding="same",
                activation=activation,
            ),
            kl.BatchNormalization(),
            kl.Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                strides=1,
                padding="same",
            ),
            kl.BatchNormalization(),
        ]
        self.act2 = kl.Activation(activation)

    def call(self, x):
        x1 = x
        for layer in self.res_layers:
            x1 = layer(x1)
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
