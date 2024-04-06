from typing import Optional, Tuple

from tensorflow import keras

kl = keras.layers


class KerasModelAddedSummary(keras.Model):
    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(input_shape)

    def init_model_graph(self, name: str = ""):
        for h in self.layers:
            if hasattr(h, "init_model_graph"):
                h.init_model_graph()

        if isinstance(self.__input_shape, list):
            x = [kl.Input(s[1:]) for s in self.__input_shape]
        else:
            x = kl.Input(self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        return keras.Model(inputs=x, outputs=self.call(x), name=name)

    def summary(self, name="", expand_nested: bool = True, **kwargs):
        model = self.init_model_graph(name=name)
        model.summary(expand_nested=expand_nested, **kwargs)
