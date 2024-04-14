import pytest

import srl
import srl.runner


def test_gym_retro():
    pytest.importorskip("retro")

    import retro  # type: ignore  # pip install gym-retro

    env_config = srl.EnvConfig("Airstriker-Genesis", gym_make_func=retro.make)

    runner = srl.Runner(env_config)
    runner.render_window()
