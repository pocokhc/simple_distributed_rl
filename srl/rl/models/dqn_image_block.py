import tensorflow.keras as keras
from tensorflow.keras import layers as kl


class DQNImageBlock(keras.Model):
    def __init__(self):
        super().__init__()

        self.image_layers = [
            kl.Conv2D(32, (8, 8), strides=(4, 4), padding="same", activation="relu"),
            kl.Conv2D(64, (4, 4), strides=(2, 2), padding="same", activation="relu"),
            kl.Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu"),
        ]

    def call(self, x):
        for layer in self.image_layers:
            x = layer(x)
        return x
