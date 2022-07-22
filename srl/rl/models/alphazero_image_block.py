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
    ):
        super().__init__()

        conv2d_params = dict(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            use_bias=False,
            activation="linear",
            kernel_regularizer=regularizers.l2(l2),
        )

        self.conv1 = kl.Conv2D(**conv2d_params)
        self.bn1 = kl.BatchNormalization()
        self.act1 = kl.LeakyReLU()
        self.resblocks = [_ResidualBlock(conv2d_params) for _ in range(n_blocks)]

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        for resblock in self.resblocks:
            x = resblock(x)
        return x


class _ResidualBlock(keras.Model):
    def __init__(self, conv2d_params):
        super().__init__()

        self.conv1 = kl.Conv2D(**conv2d_params)
        self.bn1 = kl.BatchNormalization()
        self.relu1 = kl.LeakyReLU()
        self.conv2 = kl.Conv2D(**conv2d_params)
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
    in_ = c = keras.Input((96, 72, 3))
    c = AlphaZeroImageBlock(n_blocks=1)(c)
    model = keras.Model(in_, c)
    model.summary(expand_nested=True)

    # from tensorflow.keras.utils import plot_model
    # plot_model(model, "tmp/model.png", show_shapes=True, expand_nested=True, show_layer_activations=True)
