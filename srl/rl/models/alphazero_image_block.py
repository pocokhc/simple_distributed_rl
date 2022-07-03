import tensorflow.keras as keras
from tensorflow.keras import layers as kl

"""
https://www.deepmind.com/blog/alphazero-shedding-new-light-on-chess-shogi-and-go
"""


class AlphaZeroImageBlock(kl.Layer):
    def __init__(self, n_blocks=19, filters=256):
        super().__init__()

        self.conv1 = kl.Conv2D(filters, kernel_size=3, padding="same")
        self.bn1 = kl.BatchNormalization()
        self.relu1 = kl.Activation("relu")
        self.resblocks = [_ResBlock(filters) for _ in range(n_blocks)]

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        for resblock in self.resblocks:
            x = resblock(x)
        return x


class _ResBlock(kl.Layer):
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
        x1 = self.bn2(x1)
        x = x + x1

        x = self.relu2(x)
        return x


if __name__ == "__main__":
    in_ = c = kl.Input((96, 72, 3))
    c = AlphaZeroImageBlock()(c)
    model = keras.Model(in_, c)
    model.summary(expand_nested=True)

    # from tensorflow.keras.utils import plot_model

    # plot_model(model, "tmp/model.png", show_shapes=True, expand_nested=True)
