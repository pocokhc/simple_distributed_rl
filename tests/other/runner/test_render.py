import pytest

import srl
from srl.envs import grid  # noqa E401
from srl.runner import sequence


def test_play():
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    pytest.importorskip("pygame")

    config = sequence.Config(srl.EnvConfig("Grid"), None)
    render = sequence.animation(config, max_steps=10)
    render.create_anime(draw_info=True).save("tmp/a.gif")

    config = sequence.Config(srl.EnvConfig("Grid"), None)
    render = sequence.animation(config, max_steps=10)
    render.create_anime(draw_info=True).save("tmp/b.gif")


def test_gym():
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    pytest.importorskip("pygame")
    pytest.importorskip("gym")

    config = sequence.Config(srl.EnvConfig("MountainCar-v0"), None)
    render = sequence.animation(config, max_steps=10)
    render.create_anime(draw_info=True).save("tmp/c.gif")

    config = sequence.Config(srl.EnvConfig("MountainCar-v0"), None)
    render = sequence.animation(config, max_steps=10)
    render.create_anime(draw_info=True).save("tmp/d.gif")
