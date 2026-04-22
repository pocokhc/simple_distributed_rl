from dataclasses import dataclass

import numpy as np
import pytest

from srl.base.rl.config import DummyRLConfig

pytest.importorskip("tensorflow")


@dataclass
class TestConfig(DummyRLConfig):
    a: int = 10


def test_dtype():
    cfg = TestConfig(dtype="float32")

    for fw in ["np", "numpy"]:
        np_dtype = cfg.get_dtype(fw)  # type: ignore
        assert np_dtype == np.float32

    import tensorflow as tf

    for fw in ["tf", "tensotflow"]:
        tf_dtype = cfg.get_dtype(fw)  # type: ignore
        assert tf_dtype == tf.float32
