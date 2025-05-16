from dataclasses import dataclass

import numpy as np

import srl
from srl.base.define import SpaceTypes
from srl.base.rl.config import DummyRLConfig
from srl.utils import common

"""
Spaceのテストはtest_worker_run
Space後の値や数値は各spaceのテスト
"""


@dataclass
class TestConfig(DummyRLConfig):
    pass


def test_dtype():
    cfg = TestConfig(dtype="float32")

    for fw in ["np", "numpy"]:
        np_dtype = cfg.get_dtype(fw)  # type: ignore
        assert np_dtype == np.float32

    if common.is_package_installed("torch"):
        import torch

        for fw in ["torch"]:
            torch_dtype = cfg.get_dtype(fw)  # type: ignore
            assert torch_dtype == torch.float32

    if common.is_package_installed("tensorflow"):
        import tensorflow as tf

        for fw in ["tf", "tensotflow"]:
            tf_dtype = cfg.get_dtype(fw)  # type: ignore
            assert tf_dtype == tf.float32


def test_processor():
    from srl.algorithms import ql
    from srl.rl.processors.image_processor import ImageProcessor

    rl_config = ql.Config()
    rl_config.window_length = 4
    rl_config.observation_mode = "render_image"
    rl_config.processors = [
        ImageProcessor(
            image_type=SpaceTypes.GRAY_2ch,
            resize=(84, 84),
            normalize_type="0to1",
        )
    ]
    rl_config.setup(srl.make_env("Grid"))

    assert len(rl_config.get_applied_processors()) == 1
    assert isinstance(rl_config.get_applied_processors()[0], ImageProcessor)


def test_copy():
    config = TestConfig()
    config.window_length = 4

    assert not config.is_setup()
    config.setup(srl.make_env("Grid"))
    config.window_length = 2  # changeableな変数を書き換えると、書き換えれるけどwarningが表示
    assert config.is_setup()
    config.setup(srl.make_env("Grid"))  # 2回目はsetupされない
    assert config.is_setup()

    config2 = config.copy()
    assert config2.is_setup()
    assert config2.window_length == 2


def test_summary():
    config = TestConfig(window_length=4, observation_mode="render_image")
    config.summary()
    config.setup(srl.make_env("Grid"))
    config.summary()
