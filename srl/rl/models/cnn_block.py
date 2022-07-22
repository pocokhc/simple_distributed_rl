import tensorflow.keras as keras
from tensorflow.keras import layers as kl


class SimpleCNNBlock(keras.Model):
    def __init__(
        self,
        filters: int = 256,
        use_bias: bool = False,
    ):
        super().__init__()

        conv2d_params = dict(
            use_bias=use_bias,
        )

        self.conv1 = kl.Conv2D(filters, (3, 3), padding="same", **conv2d_params)
        self.bn1 = kl.BatchNormalization()
        self.act1 = kl.LeakyReLU()

        self.conv2 = kl.Conv2D(filters, (3, 3), padding="same", **conv2d_params)
        self.bn2 = kl.BatchNormalization()
        self.act2 = kl.LeakyReLU()

        self.conv3 = kl.Conv2D(filters, (3, 3), padding="valid", **conv2d_params)
        self.bn3 = kl.BatchNormalization()
        self.act3 = kl.LeakyReLU()

        self.conv4 = kl.Conv2D(filters, (3, 3), padding="valid", **conv2d_params)
        self.bn4 = kl.BatchNormalization()
        self.act4 = kl.LeakyReLU()

    def call(self, x, training=False):
        x = self.act1(self.bn1(self.conv1(x), training=training))
        x = self.act2(self.bn2(self.conv2(x), training=training))
        x = self.act3(self.bn3(self.conv3(x), training=training))
        x = self.act4(self.bn4(self.conv4(x), training=training))
        return x


if __name__ == "__main__":
    in_ = c = keras.Input((96, 72, 3))
    c = SimpleCNNBlock()(c)
    model = keras.Model(in_, c)
    model.summary(expand_nested=True)

    # from tensorflow.keras.utils import plot_model
    # plot_model(model, "tmp/model.png", show_shapes=True, expand_nested=True, show_layer_activations=True)
