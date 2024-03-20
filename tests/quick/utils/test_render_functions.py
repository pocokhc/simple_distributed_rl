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
    print(text2)
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
        import cv2

        cv2.imshow("image", rgb_array)
        cv2.waitKey()


@pytest.mark.timeout(10)
def test_text_to_rgb_array_long():
    pytest.importorskip("PIL")

    s = ""
    for i in range(100):
        for j in range(1000):
            s += str(j)
        s += "\n"

    rgb_array = text_to_rgb_array(s)
    assert len(rgb_array.shape) == 3
    assert rgb_array.shape[2] == 3
    assert (rgb_array >= 0).all()
    assert (rgb_array <= 255).all()
    assert rgb_array.dtype == np.uint8

    # debug
    if False:
        import cv2

        cv2.imshow("image", rgb_array)
        cv2.waitKey()
