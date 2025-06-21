import logging
import os
import time
from typing import Optional

import numpy as np

from srl.base.define import RenderModeType
from srl.utils.common import is_packages_installed
from srl.utils.render_functions import print_to_text, text_to_rgb_array

logger = logging.getLogger(__name__)


class IRender:
    def render_terminal(self, **kwargs) -> None:
        pass

    def render_rgb_array(self, **kwargs) -> Optional[np.ndarray]:
        return None


class Render:
    def __init__(self, render_obj: IRender, interval: float = 1000 / 60) -> None:
        self._render_obj = render_obj
        self._mode: RenderModeType = ""
        self.rendering: bool = False
        self._screen = None

        self.set_render_options(interval=interval)
        self.cache_reset()

    def set_render_mode(self, mode: RenderModeType):
        self._mode = mode
        self.rendering = mode != ""

        if self._mode in ["rgb_array", "window"]:
            assert is_packages_installed(["PIL", "pygame"]), "This run requires installation of 'PIL', 'pygame'. (pip install pillow pygame)"
            # PIL use 'text_to_rgb_array'

    def set_render_options(
        self,
        interval: float = -1,  # ms
        scale: float = 1.0,
        font_name: str = "",
        font_size: int = 18,
    ):
        if interval > 0:
            self.interval = interval
        self.scale = scale
        self.font_name = font_name
        self.font_size = font_size

    def cache_reset(self):
        self._cache_text = ""
        self._cache_rgb_array = None

    def cache_render(self, **kwargs):
        self._cache_text = ""
        self._cache_rgb_array = None
        if self._mode in ["terminal", "terminal_rgb_array"]:
            self.get_cached_terminal_text(**kwargs)
        if self._mode in ["rgb_array", "terminal_rgb_array", "window"]:
            self.get_cached_rgb_array(**kwargs)

    def get_cached_terminal_text(self, **kwargs) -> str:
        if self._cache_text == "":
            self._cache_text = print_to_text(lambda: self._render_obj.render_terminal(**kwargs))
        return self._cache_text

    def get_cached_terminal_text_to_image(self, **kwargs) -> Optional[np.ndarray]:
        text = self.get_cached_terminal_text(**kwargs)
        if text.strip() == "":
            return None
        text_img = text_to_rgb_array(text, self.font_name, self.font_size)

        if (text_img is not None) and (self.scale != 1.0):
            import cv2

            w = int(text_img.shape[1] * self.scale)
            h = int(text_img.shape[0] * self.scale)
            text_img = cv2.resize(text_img, (w, h))

        return text_img

    def get_cached_rgb_array(self, **kwargs) -> Optional[np.ndarray]:
        if self._cache_rgb_array is None:
            self._cache_rgb_array = self._render_obj.render_rgb_array(**kwargs)
            # (H,W,C)
            if self._cache_rgb_array is not None:
                self._cache_rgb_array = self._cache_rgb_array.astype(np.uint8)

        if (self._cache_rgb_array is not None) and (self.scale != 1.0):
            import cv2

            w = int(self._cache_rgb_array.shape[1] * self.scale)
            h = int(self._cache_rgb_array.shape[0] * self.scale)
            self._cache_rgb_array = cv2.resize(self._cache_rgb_array, (w, h))

        return self._cache_rgb_array

    def render(self, **kwargs):
        if self._mode == "terminal":
            self.render_terminal(**kwargs)
        elif self._mode == "rgb_array":
            return self.get_cached_rgb_array(**kwargs)
        elif self._mode == "window":
            return self.render_window(**kwargs)

    def render_terminal(self, **kwargs):
        if self._cache_text == "":
            self._render_obj.render_terminal(**kwargs)
        else:
            print(self._cache_text)

    def render_window(self, **kwargs):
        rgb_array = self.get_cached_rgb_array(**kwargs)
        if rgb_array is None:
            rgb_array = self.get_cached_terminal_text_to_image(**kwargs)
        if rgb_array is not None:
            self._render_window_sub(rgb_array, **kwargs)
        return rgb_array

    def _render_window_sub(self, rgb_array, **kwargs):
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
