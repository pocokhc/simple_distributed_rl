from dataclasses import dataclass

import numpy as np
import pytest

from srl.base.rl.config import DummyRLConfig

pytest.importorskip("torch")
import torch


@dataclass
class TestConfig(DummyRLConfig):
    a: int = 10


def test_dtype():
    cfg = TestConfig(dtype="float32")

    for fw in ["np", "numpy"]:
        np_dtype = cfg.get_dtype(fw)  # type: ignore
        assert np_dtype == np.float32

    for fw in ["torch"]:
        torch_dtype = cfg.get_dtype(fw)  # type: ignore
        assert torch_dtype == torch.float32
