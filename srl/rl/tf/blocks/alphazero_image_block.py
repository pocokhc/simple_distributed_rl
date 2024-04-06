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
        enable_rnn: bool = False,
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

    def call(self, x, training=False):
        for layer in self.input_layers:
            x = layer(x, training=training)
        for resblock in self.resblocks:
            x = resblock(x, training=training)
        return x


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

    def call(self, x, training=False):
        x1 = x
        for layer in self.res_layers:
            x1 = layer(x1, training=training)
        x = x + x1
        x = self.act2(x)
        return x


if __name__ == "__main__":
    m = AlphaZeroImageBlock(n_blocks=1)
    m.build((None, 96, 72, 3))
    m.summary(expand_nested=True)
