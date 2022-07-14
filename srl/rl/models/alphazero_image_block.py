import tensorflow.keras as keras
from tensorflow.keras import layers as kl

"""
https://www.deepmind.com/blog/alphazero-shedding-new-light-on-chess-shogi-and-go
https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
"""


class AlphaZeroImageBlock(keras.Model):
    def __init__(
        self,
        n_blocks: int = 19,
        filters: int = 256,
    ):
        super().__init__()

        self.conv1 = kl.Conv2D(filters, kernel_size=3, padding="same")
        self.bn1 = kl.BatchNormalization()
        self.act1 = kl.Activation("relu")
        self.resblocks = [_ResidualBlock(filters) for _ in range(n_blocks)]

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        for resblock in self.resblocks:
            x = resblock(x)
        return x


class _ResidualBlock(keras.Model):
    def __init__(self, filters):
        super().__init__()

        self.conv1 = kl.Conv2D(filters, kernel_size=3, padding="same")
        self.bn1 = kl.BatchNormalization()
        self.relu1 = kl.Activation("relu")
        self.conv2 = kl.Conv2D(filters, kernel_size=3, padding="same")
        self.bn2 = kl.BatchNormalization()
        self.relu2 = kl.Activation("relu")

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
