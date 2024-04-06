from tensorflow import keras

kl = keras.layers


class DQNImageBlock(keras.Model):
    def __init__(
        self,
        filters: int = 32,
        activation: str = "relu",
        enable_rnn: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_layers = [
            kl.Conv2D(filters, (8, 8), strides=(4, 4), padding="same", activation=activation),
            kl.Conv2D(filters * 2, (4, 4), strides=(2, 2), padding="same", activation=activation),
            kl.Conv2D(filters * 2, (3, 3), strides=(1, 1), padding="same", activation=activation),
        ]

        # Conv2Dはshape[-3:]を処理するのでTimeDistributedは不要
        if enable_rnn:
            self.image_layers = [kl.TimeDistributed(x) for x in self.image_layers]

    def call(self, x, training=False):
        for layer in self.image_layers:
            x = layer(x, training=training)
        return x


if __name__ == "__main__":
    m = DQNImageBlock()
    m.build((None, None, 64, 75, 19))
    m.summary()
