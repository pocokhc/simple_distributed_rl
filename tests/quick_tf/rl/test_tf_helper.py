import numpy as np
import pytest

pytest.importorskip("tensorflow")

from tensorflow import keras

from srl.rl.tf.helper import model_soft_sync


def _create_simple_model() -> keras.Model:
    """簡単なSequentialモデルを作成"""
    model = keras.Sequential([keras.layers.Dense(2, input_shape=(2,), use_bias=True)])
    return model


def test_model_soft_sync():
    source_model = _create_simple_model()
    target_model = _create_simple_model()

    # 重みを明示的に設定（source_model の方を 1.0 固定、target_model は 0.0 固定）
    source_weights = [np.ones_like(w) for w in source_model.get_weights()]
    target_weights = [np.full_like(w, 0.5) for w in target_model.get_weights()]
    source_model.set_weights(source_weights)
    target_model.set_weights(target_weights)

    model_soft_sync(target_model, source_model, tau=0.1)

    # 実際の重みと比較
    new_source_weights = source_model.get_weights()
    new_target_weights = target_model.get_weights()
    print(new_source_weights)
    print(new_target_weights)
    assert (new_source_weights[0] == np.full((2, 2), 1, np.float32)).all()
    assert (new_target_weights[0] == np.full((2, 2), 0.55, np.float32)).all()
    assert (new_source_weights[1] == np.full((2,), 1, np.float32)).all()
    assert (new_target_weights[1] == np.full((2,), 0.55, np.float32)).all()
    assert len(new_source_weights) == 2
    assert len(new_target_weights) == 2
