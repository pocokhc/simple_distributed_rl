import numpy as np
import pytest

import srl.rl.dummy
from srl.base.define import EnvObservationTypes
from srl.base.spaces.box import BoxSpace
from srl.rl.processors.render_image_processor import RenderImageProcessor
from srl.test.processor import TestProcessor


def test_image():
    pytest.importorskip("pygame")

    tester = TestProcessor()
    processor = RenderImageProcessor()

    env_name = "Grid"

    tester.run(processor, env_name)
    tester.preprocess_observation_space(
        processor,
        env_name,
        EnvObservationTypes.COLOR,
        BoxSpace((160, 192, 3), 0, 255),
    )

    in_image = np.ones((3,)).astype(int)
    out_image = np.ones((160, 192, 3)).astype(int)
    tester.preprocess_observation(
        processor,
        env_name,
        in_observation=in_image,
        out_observation=out_image,
        check_shape_only=True,
    )
