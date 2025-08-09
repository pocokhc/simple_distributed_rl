import logging
import traceback

from tensorflow import keras

logger = logging.getLogger(__name__)

kl = keras.layers


class KerasModelAddedSummary(keras.Model):
    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(input_shape)

    def init_model_graph(self, name: str = ""):
        try:
            if not hasattr(self, "_KerasModelAddedSummary__input_shape"):
                return None

            for h in self.layers:
                if hasattr(h, "init_model_graph"):
                    h.init_model_graph()

            if isinstance(self.__input_shape, list):
                x = [
                    [kl.Input(s2[1:]) for s2 in s] if isinstance(s, list) else kl.Input(s[1:])
                    for s in self.__input_shape  #
                ]
            else:
                x = kl.Input(self.__input_shape[1:])
            name = self.__class__.__name__ if name == "" else name
            model = keras.Model(inputs=x, outputs=self.call(x), name=name)
        except Exception:
            logger.warning(traceback.format_exc())
            return None
        return model

    def summary(self, name="", expand_nested: bool = True, **kwargs):
        model = self.init_model_graph(name=name)
        if model is None:
            super().summary(expand_nested=expand_nested, **kwargs)
        else:
            model.summary(expand_nested=expand_nested, **kwargs)


class SequentialModel(KerasModelAddedSummary):
    def __init__(self, layers: list, **kwargs):
        super().__init__(**kwargs)
        self.h_layers = layers

    def call(self, x, training=False):
        for h in self.h_layers:
            x = h(x, training=training)
        return x
