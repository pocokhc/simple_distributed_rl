import os

import pytest

import srl
from srl.algorithms import ql
from srl.base.define import SpaceTypes
from srl.rl.processors.image_processor import ImageProcessor
from srl.utils import common


def test_1(tmp_path):
    pytest.importorskip("pygame")

    common.logger_print()

    rl_config = ql.Config()
    rl_config.window_length = 4
    rl_config.observation_mode = "image"
    rl_config.processors = [
        ImageProcessor(
            image_type=SpaceTypes.GRAY_2ch,
            resize=(84, 84),
            enable_norm=True,
        )
    ]
    runner = srl.Runner("Grid", rl_config)

    path = os.path.join(tmp_path, "a.gif")
    runner.animation_save_gif(path)
