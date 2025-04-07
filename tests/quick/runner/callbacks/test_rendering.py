import os

import pytest

import srl
from srl.algorithms import ql
from srl.base.define import SpaceTypes
from srl.rl.processors.image_processor import ImageProcessor


def test_1(tmp_path):
    pytest.importorskip("pygame")
    env_config = srl.EnvConfig(
        "Grid",
        processors=[
            ImageProcessor(
                image_type=SpaceTypes.GRAY_2ch,
                resize=(84, 84),
                normalize_type="0to1",
            )
        ],
    )

    rl_config = ql.Config()
    rl_config.window_length = 4
    rl_config.observation_mode = "render_image"
    rl_config.processors = []
    runner = srl.Runner(env_config, rl_config)

    path = os.path.join(tmp_path, "a.gif")
    runner.animation_save_gif(path)
