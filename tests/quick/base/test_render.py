from typing import Optional

import numpy as np
import pytest

from srl.base.render import IRender, Render
from srl.utils.common import is_available_pygame_video_device


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
    render.set_render_mode(mode="terminal")

    text = "StubRender\nAAA"
    render.render(text=text)

    text2 = render.get_cached_terminal_text(text=text)
    assert text2 == text + "\n"


def test_render_terminal_to_image():
    render = Render(StubRender())
    render.set_render_mode(mode="rgb_array")

    text = "StubRender\nAAA"
    img = render.get_cached_terminal_text_to_image(text=text)
    assert img is not None
    assert len(img.shape) == 3
    assert img.shape[2] == 3
    assert (img >= 0).all()
    assert (img <= 255).all()
    assert img.dtype == np.uint8


@pytest.mark.parametrize(
    "return_rgb",
    [
        False,
        True,
    ],
)
def test_render_rgb_array(return_rgb):
    render = Render(StubRender())
    render.set_render_mode(mode="terminal")

    text = "StubRender\nAAA"

    img = render.get_cached_rgb_array(return_rgb=return_rgb, text=text)
    if return_rgb:
        assert img is not None
        assert len(img.shape) == 3
        assert img.shape[2] == 3
        assert (img >= 0).all()
        assert (img <= 255).all()
        assert img.dtype == np.uint8
    else:
        assert img is None


def test_render_window():
    if not is_available_pygame_video_device():
        pytest.skip("pygame.error: No available video device")

    render = Render(StubRender())
    render.set_render_mode(mode="window")

    for _ in range(10):
        render.render(return_rgb=True)
