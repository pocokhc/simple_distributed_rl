import logging
import os
import time
from typing import Optional, Union

import numpy as np

from srl.base.define import PlayRenderModes, RenderModes
from srl.utils.common import is_packages_installed
from srl.utils.render_functions import print_to_text, text_to_rgb_array

logger = logging.getLogger(__name__)


class IRender:
    def set_render_mode(self, mode: RenderModes) -> None:
        pass

    def render_terminal(self, **kwargs) -> None:
        pass

    def render_rgb_array(self, **kwargs) -> Optional[np.ndarray]:
        return None


class Render:
    def __init__(self, render_obj: IRender) -> None:
        self.render_obj = render_obj
        self.mode = PlayRenderModes.none

        self.interval: float = -1  # ms
        self.scale: float = 1.0
        self.font_name: str = ""
        self.font_size: int = 12

        self.print_str = ""
        self.rgb_array = None

        self.screen = None

    def reset(self, mode: Union[str, PlayRenderModes]):
        self.mode = PlayRenderModes.from_str(mode)

        if self.mode in [PlayRenderModes.rgb_array, PlayRenderModes.window]:
            assert is_packages_installed(
                ["PIL", "pygame"]
            ), "This run requires installation of 'PIL', 'pygame'. (pip install pillow pygame)"
            # PIL use 'text_to_rgb_array'

        self.render_obj.set_render_mode(PlayRenderModes.convert_render_mode(self.mode))

    def cache_reset(self):
        self.print_str = ""
        self.rgb_array = None

    # ----------------------------

    def get_dummy(self) -> Union[None, str, np.ndarray]:
        if self.mode == PlayRenderModes.none:
            return
        elif self.mode == PlayRenderModes.terminal:
            return
        elif self.mode == PlayRenderModes.ansi:
            return ""
        elif self.mode == PlayRenderModes.rgb_array:
            return np.zeros((4, 4, 3), dtype=np.uint8)
        elif self.mode == PlayRenderModes.window:
            return

    def render(self, **kwargs) -> Union[None, str, np.ndarray]:
        if self.mode == PlayRenderModes.none:
            return
        elif self.mode == PlayRenderModes.terminal:
            return self.render_terminal(**kwargs)
        elif self.mode == PlayRenderModes.ansi:
            return self.render_terminal(return_text=True, **kwargs)
        elif self.mode == PlayRenderModes.rgb_array:
            return self.render_rgb_array(**kwargs)
        elif self.mode == PlayRenderModes.window:
            return self.render_window(**kwargs)

    def render_terminal(self, return_text: bool = False, **kwargs) -> Union[None, str]:
        if return_text:
            if self.print_str == "":
                self.print_str = print_to_text(lambda: self.render_obj.render_terminal(**kwargs))
            return self.print_str
        else:
            self.render_obj.render_terminal(**kwargs)

    def render_rgb_array(self, **kwargs) -> np.ndarray:
        if self.rgb_array is None:
            self.rgb_array = self.render_obj.render_rgb_array(**kwargs)
        if self.rgb_array is None:
            text = print_to_text(lambda: self.render_obj.render_terminal(**kwargs))
            if text == "":
                return np.zeros((4, 4, 3), dtype=np.uint8)  # dummy
            self.rgb_array = text_to_rgb_array(text, self.font_name, self.font_size)
        return self.rgb_array.astype(np.uint8)

    def render_window(self, **kwargs) -> np.ndarray:
        rgb_array = self.render_rgb_array(**kwargs)

        import pygame

        from srl.utils import pygame_wrapper as pw

        if self.screen is None:
            if "SDL_VIDEODRIVER" in os.environ:
                pygame.display.quit()
                del os.environ["SDL_VIDEODRIVER"]

            pygame.init()
            w = int(rgb_array.shape[1] * self.scale)
            h = int(rgb_array.shape[0] * self.scale)

            w = min(w, 1900)
            h = min(h, 1600)
            self.screen = pygame.display.set_mode((w, h))
            self.t0 = time.time()

        pw.draw_image_rgb_array(self.screen, 0, 0, rgb_array)
        pygame.display.flip()

        # --- interval loop
        while True:
            pygame.event.get()

            if self.interval <= 0:
                break
            elapsed_time = time.time() - self.t0
            if elapsed_time > self.interval / 1000:
                break
        self.t0 = time.time()

        return rgb_array
