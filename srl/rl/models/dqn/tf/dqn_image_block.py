from tensorflow import keras

kl = keras.layers


class DQNImageBlock(keras.Model):
    def __init__(
        self,
        filters: int = 32,
        activation="relu",
        enable_time_distributed_layer: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_layers = [
            kl.Conv2D(filters, (8, 8), strides=(4, 4), padding="same", activation=activation),
            kl.Conv2D(filters * 2, (4, 4), strides=(2, 2), padding="same", activation=activation),
            kl.Conv2D(filters * 2, (3, 3), strides=(1, 1), padding="same", activation=activation),
        ]

        if enable_time_distributed_layer:
            self.image_layers = [kl.TimeDistributed(x) for x in self.image_layers]

    def call(self, x, training=False):
        for layer in self.image_layers:
            x = layer(x, training=training)
        return x

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def init_model_graph(self, name: str = ""):
        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        keras.Model(inputs=x, outputs=self.call(x), name=name)


if __name__ == "__main__":
    m = DQNImageBlock()
    m.build((None, 64, 75, 19))
    m.init_model_graph()
    m.summary()
