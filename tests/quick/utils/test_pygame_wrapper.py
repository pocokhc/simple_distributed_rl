import os

import numpy as np
import pytest


def test_render():
    pytest.importorskip("pygame")

    from srl.utils import pygame_wrapper as pw

    screen = pw.create_surface(640, 480)

    pw.draw_fill(screen)
    pw.draw_line(screen, 0, 0, 100, 100)
    pw.draw_box(screen, 10, 10, 50, 50, filled=True)
    pw.draw_circle(screen, 100, 10, 10, filled=False)
    pw.draw_circle(screen, 100, 100, 10, filled=True)
    pw.draw_text(screen, 50, 200, "あいうえお")
    pw.draw_texts(screen, 50, 250, ["あいうえお", "おはよう"])

    rgb_array = np.full((5, 5, 3), 128, dtype=np.uint8)
    pw.draw_image_rgb_array(screen, 0, 200, rgb_array, resize=(10, 50))

    img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../srl/envs/img/cell.png"))
    pw.load_image("A", img_path)
    pw.draw_image(screen, "A", 300, 10)

    rgb_array = pw.get_rgb_array(screen)

    # debug
    if False:
        import cv2

        cv2.imshow("image", rgb_array)
        cv2.waitKey()
