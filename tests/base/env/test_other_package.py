import pytest

import srl


def test_gym_retro():
    pytest.importorskip("retro")

    import retro  # type: ignore  # pip install gym-retro

    env_config = srl.EnvConfig("Airstriker-Genesis", gym_make_func=retro.make)

    env = srl.make_env(env_config)
    env.reset()
    while not env.done:
        env.step(env.sample())
    env.close()
