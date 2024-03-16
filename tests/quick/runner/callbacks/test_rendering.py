import os

import pytest

import srl
from srl.algorithms import ql
from srl.base.define import EnvTypes
from srl.rl.processors.image_processor import ImageProcessor
from srl.utils import common


def test_1():
    pytest.importorskip("pygame")

    common.logger_print()

    rl_config = ql.Config()
    rl_config.window_length = 4
    rl_config.use_render_image_for_observation = True
    rl_config.processors = [
        ImageProcessor(
            image_type=EnvTypes.GRAY_2ch,
            resize=(84, 84),
            enable_norm=True,
        )
    ]
    runner = srl.Runner("Grid", rl_config)

    os.makedirs("tmp_test", exist_ok=True)

    path = os.path.join("tmp_test", "a.gif")
    runner.animation_save_gif(path)
