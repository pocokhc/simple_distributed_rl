import tensorflow.keras as keras
from tensorflow.keras import layers as kl

"""
R2D3 Image Model
https://arxiv.org/abs/1909.01387
"""


class R2D3ImageBlock(keras.Model):
    def __init__(self, enable_time_distributed_layer: bool = False):
        super().__init__()

        self.res1 = _ResBlock(16, enable_time_distributed_layer)
        self.res2 = _ResBlock(32, enable_time_distributed_layer)
        self.res3 = _ResBlock(32, enable_time_distributed_layer)
        self.relu = kl.Activation("relu")
        self.flatten = kl.Flatten()
        self.dense = kl.Dense(256, activation="relu")

        if enable_time_distributed_layer:
            self.flatten = kl.TimeDistributed(self.flatten)
            self.dense = kl.TimeDistributed(self.dense)

    def call(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def init_model_graph(self, name: str = ""):
        self.res1.init_model_graph()
        self.res2.init_model_graph()
        self.res3.init_model_graph()

        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        keras.Model(inputs=x, outputs=self.call(x), name=name)


class _ResBlock(keras.Model):
    def __init__(self, n_filter, enable_time_distributed_layer):
        super().__init__()

        self.conv = kl.Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same")
        self.pool = kl.MaxPooling2D((3, 3), strides=(2, 2), padding="same")
        self.res1 = _ResidualBlock(n_filter)
        self.res2 = _ResidualBlock(n_filter)

        if enable_time_distributed_layer:
            self.pool = kl.TimeDistributed(self.pool)

    def call(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def init_model_graph(self, name: str = ""):
        self.res1.init_model_graph()
        self.res2.init_model_graph()

        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        keras.Model(inputs=x, outputs=self.call(x), name=name)


class _ResidualBlock(keras.Model):
    def __init__(self, n_filter):
        super().__init__()

        self.relu1 = kl.Activation("relu")
        self.conv1 = kl.Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same")
        self.relu2 = kl.Activation("relu")
        self.conv2 = kl.Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same")
        self.add = kl.Add()

    def call(self, x):
        x1 = self.relu1(x)
        x1 = self.conv1(x1)
        x1 = self.relu2(x1)
        x1 = self.conv2(x1)
        return self.add([x, x1])

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def init_model_graph(self, name: str = ""):
        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        keras.Model(inputs=x, outputs=self.call(x), name=name)


if __name__ == "__main__":
    m = R2D3ImageBlock(enable_time_distributed_layer=True)
    m.build((None, None, 96, 72, 3))
    m.init_model_graph()
    m.summary(expand_nested=True)

    # from tensorflow.keras.utils import plot_model

    # plot_model(model, "tmp/model.png", show_shapes=True, expand_nested=True)
