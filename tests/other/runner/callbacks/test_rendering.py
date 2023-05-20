import pytest

from srl import runner
from srl.algorithms import ql
from srl.base.define import PlayRenderMode
from srl.envs import grid  # noqa F401
from srl.runner.callbacks.rendering import Rendering
from srl.runner.core import play
from srl.utils import common

common.logger_print()


@pytest.mark.parametrize(
    "mode",
    [
        PlayRenderMode.none,
        PlayRenderMode.terminal,
        PlayRenderMode.ansi,
        PlayRenderMode.window,
        PlayRenderMode.rgb_array,
    ],
)
def test_run(mode):
    if mode == PlayRenderMode.window or mode == PlayRenderMode.rgb_array:
        pytest.importorskip("cv2")
        pytest.importorskip("matplotlib")
        pytest.importorskip("PIL")
        pytest.importorskip("pygame")

    config = runner.Config("Grid", ql.Config())
    callback = Rendering(mode)
    play(
        config,
        max_episodes=1,
        train_only=False,
        enable_profiling=False,
        training=False,
        eval=None,
        history=None,
        checkpoint=None,
        callbacks=[callback],
    )


def test_anime():
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    pytest.importorskip("pygame")

    config = runner.Config("Grid", ql.Config())
    callback = Rendering(PlayRenderMode.rgb_array)
    play(
        config,
        max_episodes=1,
        train_only=False,
        enable_profiling=False,
        training=False,
        eval=None,
        history=None,
        checkpoint=None,
        callbacks=[callback],
    )
    callback.create_anime(draw_info=True)
