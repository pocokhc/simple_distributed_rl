import numpy as np
import pytest

from srl.utils.render_functions import print_to_text, text_to_rgb_array


@pytest.mark.parametrize(
    "name, text",
    [
        ["str", "StubRender\nAAA"],
        ["none", ""],
        ["japanese", "あいうえお"],
    ],
)
def test_print_to_text(name, text):
    text2 = print_to_text(lambda: print(text))
    assert text2 == text + "\n"


@pytest.mark.parametrize(
    "name, text",
    [
        ["str", "StubRender\nAAA"],
        ["none", ""],
        ["japanese", "あいうえお"],
    ],
)
def test_text_to_rgb_array(name, text):
    pytest.importorskip("PIL")

    rgb_array = text_to_rgb_array(text)
    assert len(rgb_array.shape) == 3
    assert rgb_array.shape[2] == 3
    assert (rgb_array >= 0).all()
    assert (rgb_array <= 255).all()
    assert rgb_array.dtype == np.uint8

    # debug
    if False:
        from PIL import Image

        img = Image.fromarray(rgb_array)
        img.show()
