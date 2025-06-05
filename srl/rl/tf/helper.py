from tensorflow import keras

"""
tfに関する処理を助けるライブラリ群
tfに依存していない処理はfunctionsへ
"""


def model_soft_sync(target_model: keras.Model, source_model: keras.Model, tau: float):
    new_w = [
        (1 - tau) * w_tgt + tau * w
        for w_tgt, w in zip(target_model.get_weights(), source_model.get_weights())  #
    ]
    target_model.set_weights(new_w)
