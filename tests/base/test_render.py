from typing import Optional

import numpy as np
import pytest

from srl.base.render import IRender, Render
from srl.utils.common import is_available_video_device


class StubRender(IRender):
    def render_terminal(self, text, **kwargs) -> None:
        print(text)

    def render_rgb_array(self, return_rgb, **kwargs) -> Optional[np.ndarray]:
        if return_rgb:
            return np.zeros((4, 4, 3))
        else:
            return None


def test_render_terminal():
    render = Render(StubRender())

    text = "StubRender\nAAA"
    render.render_terminal(return_text=False, text=text)

    text2 = render.render_terminal(return_text=True, text=text)
    assert text2 == text + "\n"


@pytest.mark.parametrize(
    "return_rgb",
    [
        False,
        True,
    ],
)
def test_render_rgb_array(return_rgb):
    pytest.importorskip("PIL")

    render = Render(StubRender())

    text = "StubRender\nAAA"

    rgb_array = render.render_rgb_array(return_rgb=return_rgb, text=text)
    assert len(rgb_array.shape) == 3
    assert rgb_array.shape[2] == 3
    assert (rgb_array >= 0).all()
    assert (rgb_array <= 255).all()
    assert rgb_array.dtype == np.uint8


def test_render_window():
    pytest.importorskip("PIL")
    pytest.importorskip("pygame")
    if not is_available_video_device():
        pytest.skip("pygame.error: No available video device")

    render = Render(StubRender())

    for _ in range(10):
        render.render_window(return_rgb=True)
