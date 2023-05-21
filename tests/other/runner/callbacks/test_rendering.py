import pytest

from srl import runner
from srl.algorithms import ql
from srl.base.define import PlayRenderModes
from srl.envs import grid  # noqa F401
from srl.runner.callbacks.rendering import Rendering
from srl.runner.core import play
from srl.utils import common

common.logger_print()


@pytest.mark.parametrize(
    "mode",
    [
        PlayRenderModes.none,
        PlayRenderModes.terminal,
        PlayRenderModes.ansi,
        PlayRenderModes.window,
        PlayRenderModes.rgb_array,
    ],
)
def test_run(mode):
    if mode == PlayRenderModes.window or mode == PlayRenderModes.rgb_array:
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
    callback = Rendering(PlayRenderModes.rgb_array)
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
