from typing import Tuple

from tensorflow import keras

kl = keras.layers


class MLPBlock(keras.Model):
    def __init__(
        self,
        layer_sizes: Tuple[int, ...],
        activation: str = "relu",
        use_bias=True,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        enable_time_distributed_layer: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_layers = []
        for h in layer_sizes:
            self.hidden_layers.append(
                kl.Dense(
                    h,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint,
                )
            )

        if enable_time_distributed_layer:
            self.hidden_layers = [kl.TimeDistributed(x) for x in self.hidden_layers]

    def call(self, x, training=False):
        for layer in self.hidden_layers:
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
    m = MLPBlock((512, 128, 256))
    m.build((None, 64))
    m.init_model_graph()
    m.summary()
