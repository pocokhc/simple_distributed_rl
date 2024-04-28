import numpy as np
from tensorflow import keras

"""
tfに関する処理を助けるライブラリ群
tfに依存していない処理はfunctionsへ
"""


def model_soft_sync(target_model: keras.Model, source_model: keras.Model, tau: float):
    target_model.set_weights(
        (1 - tau) * np.array(target_model.get_weights(), dtype=object)
        + tau * np.array(source_model.get_weights(), dtype=object)
    )
