import logging
import os
import time
from typing import Optional, Union

import numpy as np

from srl.base.define import RenderModes
from srl.utils.common import is_packages_installed
from srl.utils.render_functions import print_to_text, text_to_rgb_array

logger = logging.getLogger(__name__)


class IRender:
    def render_terminal(self, **kwargs) -> None:
        pass

    def render_rgb_array(self, **kwargs) -> Optional[np.ndarray]:
        return None


class Render:
    def __init__(self, render_obj: IRender) -> None:
        self._render_obj = render_obj
        self._mode = RenderModes.none

        self._cache_ansi = ""
        self._cache_rgb_array = None

        self._screen = None
        self.set_render_options()

    def set_render_mode(self, mode: Union[str, RenderModes]):
        self.cache_reset()
        self._mode = RenderModes.from_str(mode)

        if self._mode in [RenderModes.rgb_array, RenderModes.window]:
            assert is_packages_installed(
                ["PIL", "pygame"]
            ), "This run requires installation of 'PIL', 'pygame'. (pip install pillow pygame)"
            # PIL use 'text_to_rgb_array'

    def set_render_options(
        self,
        interval: float = -1,  # ms
        scale: float = 1.0,
        font_name: str = "",
        font_size: int = 18,
    ):
        self.interval = interval
        self.scale = scale
        self.font_name = font_name
        self.font_size = font_size

    def cache_reset(self):
        self.print_str = ""
        self.rgb_array = None

    def render(self, render_window: bool = True, **kwargs) -> None:
        if self._mode == RenderModes.terminal:
            self._render_obj.render_terminal(**kwargs)
        elif render_window and self._mode == RenderModes.window:
            self._render_window(**kwargs)

    def render_ansi(self, **kwargs) -> str:
        if self.print_str == "":
            self.print_str = print_to_text(lambda: self._render_obj.render_terminal(**kwargs))
        return self.print_str

    def render_rgb_array(self, **kwargs) -> np.ndarray:
        if self.rgb_array is None:
            self.rgb_array = self._render_obj.render_rgb_array(**kwargs)
            if self.scale != 1.0:
                import cv2

                w = int(self.rgb_array.shape[1] * self.scale)
                h = int(self.rgb_array.shape[0] * self.scale)
                self.rgb_array = cv2.resize(self.rgb_array, (w, h))
        if self.rgb_array is None:
            text = self.render_ansi(**kwargs)
            if text == "":
                return np.zeros((4, 4, 3), dtype=np.uint8)  # dummy
            self.rgb_array = text_to_rgb_array(text, self.font_name, self.font_size)
        return self.rgb_array.astype(np.uint8)

    def _render_window(self, **kwargs) -> np.ndarray:
        rgb_array = self.render_rgb_array(**kwargs)

        import pygame

        from srl.utils import pygame_wrapper as pw

        if self._screen is None:
            if "SDL_VIDEODRIVER" in os.environ:
                pygame.display.quit()
                del os.environ["SDL_VIDEODRIVER"]

            pygame.init()
            w = min(rgb_array.shape[1], 1200)
            h = min(rgb_array.shape[0], 900)

            logger.info(f"create pygame({w},{h}), interval {self.interval}ms")
            self._screen = pygame.display.set_mode((w, h))
            self._t0 = time.time()

        pw.draw_image_rgb_array(self._screen, 0, 0, rgb_array)
        pygame.display.flip()

        # --- interval loop
        while True:
            pygame.event.get()

            if self.interval <= 0:
                break
            elapsed_time = time.time() - self._t0
            if elapsed_time > self.interval / 1000:
                break
        self._t0 = time.time()

        return rgb_array
