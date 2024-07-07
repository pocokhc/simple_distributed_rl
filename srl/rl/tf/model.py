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
            for h in self.layers:
                if hasattr(h, "init_model_graph"):
                    h.init_model_graph()

            if isinstance(self.__input_shape, list):
                x = [kl.Input(s[1:]) for s in self.__input_shape]
            else:
                x = kl.Input(self.__input_shape[1:])
            name = self.__class__.__name__ if name == "" else name
            model = keras.Model(inputs=x, outputs=self.call(x), name=name)
        except Exception:
            logger.error(traceback.format_exc())
            return None
        return model

    def summary(self, name="", expand_nested: bool = True, **kwargs):
        model = self.init_model_graph(name=name)
        if model is None:
            super().summary(expand_nested=expand_nested, **kwargs)
        else:
            model.summary(expand_nested=expand_nested, **kwargs)
