from typing import Optional

import numpy as np
import pytest

from srl.base.renderer import IRenderer, Renderer
from srl.utils.common import is_available_pygame_video_device


class StubRender(IRenderer):
    def render_terminal(self, text, **kwargs) -> None:
        print(text)

    def render_rgb_array(self, **kwargs) -> Optional[np.ndarray]:
        return np.zeros((4, 4, 3))


def _assert_terminal_text(render, text):
    text2 = render.get_terminal_text()
    assert text2 == text + "\n"


def _assert_terminal_rgb_array(render):
    img = render.get_terminal_rgb_array()
    assert img is not None
    assert len(img.shape) == 3
    assert img.shape[2] == 3
    assert (img >= 0).all()
    assert (img <= 255).all()
    assert img.dtype == np.uint8


def _assert_rgb_array(render):
    img = render.get_rgb_array()
    assert img is not None
    assert len(img.shape) == 3
    assert img.shape[2] == 3
    assert (img >= 0).all()
    assert (img <= 255).all()
    assert img.dtype == np.uint8


def test_render_terminal():
    render = Renderer(StubRender())
    render.setup_render_mode(
        requested_render_modes={"terminal"},
        base_render_target="terminal_to_text",
        interval=0,
    )

    text = "StubRender\nAAA"
    render.update_cache(text=text)

    _assert_terminal_text(render, text)


def test_render_terminal_to_image():
    pytest.importorskip("pygame")
    pytest.importorskip("PIL")
    render = Renderer(StubRender())
    render.setup_render_mode(
        requested_render_modes={"terminal"},
        base_render_target="terminal_to_rgb_array",
        interval=0,
    )

    text = "StubRender\nAAA"
    render.update_cache(text=text)

    _assert_terminal_rgb_array(render)


def test_render_rgb_array():
    render = Renderer(StubRender())
    render.setup_render_mode(
        requested_render_modes={"rgb_array"},
        base_render_target="rgb_array",
        interval=0,
    )

    text = "StubRender\nAAA"
    render.update_cache(text=text)

    _assert_rgb_array(render)


def test_render_window():
    if not is_available_pygame_video_device():
        pytest.skip("pygame.error: No available video device")

    render = Renderer(StubRender())
    render.setup_render_mode(
        requested_render_modes={"rgb_array"},
        base_render_target="window",
        interval=0,
    )
    for _ in range(10):
        render.render()


def test_render_rl():
    render = Renderer(StubRender())
    render.setup_render_mode(
        requested_render_modes={"terminal", "rgb_array"},
        base_render_target="",
        interval=0,
    )
    text = "StubRender\nAAA"
    render.update_cache(text=text)

    _assert_terminal_text(render, text)
    _assert_terminal_rgb_array(render)
    _assert_rgb_array(render)
