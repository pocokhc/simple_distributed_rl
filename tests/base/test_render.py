from typing import Optional

import numpy as np
import pytest

from srl.base.render import IRender, Render
from srl.utils.common import is_packages_installed


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


@pytest.mark.skipif(not is_packages_installed(["cv2", "matplotlib", "PIL", "pygame"]), reason="no module")
@pytest.mark.parametrize(
    "return_rgb",
    [
        False,
        True,
    ],
)
def test_render_rgb_array(return_rgb):
    render = Render(StubRender())

    text = "StubRender\nAAA"

    rgb_array = render.render_rgb_array(return_rgb=return_rgb, text=text)
    assert len(rgb_array.shape) == 3
    assert rgb_array.shape[2] == 3
    assert (rgb_array >= 0).all()
    assert (rgb_array <= 255).all()
    assert rgb_array.dtype == np.uint8


@pytest.mark.skipif(not is_packages_installed(["cv2", "matplotlib", "PIL", "pygame"]), reason="no module")
def test_render_window():
    render = Render(StubRender())

    for _ in range(10):
        render.render_window(return_rgb=True)
