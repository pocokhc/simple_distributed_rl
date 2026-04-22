import logging
import os
import time
from typing import Optional, Set

import numpy as np

from srl.base.define import RenderTarget, SupportedRenderMode
from srl.utils.common import is_packages_installed
from srl.utils.render_functions import print_to_text, text_to_rgb_array

logger = logging.getLogger(__name__)


class IRenderer:
    def render_terminal(self, **kwargs) -> None:
        pass

    def render_rgb_array(self, **kwargs) -> Optional[np.ndarray]:
        return None


class Renderer:
    def __init__(self, render_obj: IRenderer):
        self._render_obj = render_obj
        self.requested_render_modes: Set[SupportedRenderMode] = set()
        self.base_render_target: RenderTarget = ""
        self.interval: float = 1000 / 60
        self.rendering: bool = False
        self._screen = None

        self.set_render_options()
        self.cache_clear()

    def setup_render_mode(self, requested_render_modes: Set[SupportedRenderMode], base_render_target: RenderTarget, interval: float):
        self.requested_render_modes = {s for s in requested_render_modes if s != ""}
        self.rendering = len(self.requested_render_modes) > 0
        self.base_render_target = base_render_target
        self.interval = interval

        # 整合性チェック
        if base_render_target == "":
            pass
        elif base_render_target == "terminal":
            # 直接printは複雑になるので止める
            assert "terminal" in self.requested_render_modes
        elif base_render_target == "terminal_to_text":
            assert "terminal" in self.requested_render_modes
        elif base_render_target == "terminal_to_rgb_array":
            assert "terminal" in self.requested_render_modes
        elif base_render_target == "rgb_array":
            assert "rgb_array" in self.requested_render_modes
        elif base_render_target == "window":
            assert "rgb_array" in self.requested_render_modes
        else:
            raise UnimplementedCaseError()

        if "rgb_array" in self.requested_render_modes:
            # PIL use 'text_to_rgb_array'
            assert is_packages_installed(["PIL", "pygame"]), "This run requires installation of 'PIL', 'pygame'. (pip install pillow pygame)"

    def set_render_options(
        self,
        scale: float = 1.0,
        font_name: str = "",
        font_size: int = 18,
    ):
        self.scale = scale
        self.font_name = font_name
        self.font_size = font_size

    def cache_clear(self):
        self._cache_text = ""
        self._cache_text_rgb_array = None
        self._cache_rgb_array = None

    def update_cache(self, **kwargs):
        if "rgb_array" in self.requested_render_modes:
            self.refresh_cache_rgb_array(**kwargs)
            # rgbの取得に失敗したらterminalの取得に変更
            if self._cache_rgb_array is None:
                if "terminal" not in self.requested_render_modes:
                    logger.info("Adding 'terminal' to requested_render_modes.")
                self.requested_render_modes.add("terminal")
        if "terminal" in self.requested_render_modes:
            self.refresh_cache_terminal_to_text(**kwargs)

    def refresh_cache_terminal_to_text(self, **kwargs):
        self._cache_text_rgb_array = None
        self._cache_text = print_to_text(lambda: self._render_obj.render_terminal(**kwargs))

    def refresh_cache_rgb_array(self, **kwargs):
        self._cache_rgb_array = self._render_obj.render_rgb_array(**kwargs)
        if self._cache_rgb_array is None:
            return

        # (H,W,C)
        self._cache_rgb_array = self._cache_rgb_array.astype(np.uint8)

        if self.scale != 1.0:
            import cv2

            w = int(self._cache_rgb_array.shape[1] * self.scale)
            h = int(self._cache_rgb_array.shape[0] * self.scale)
            self._cache_rgb_array = cv2.resize(self._cache_rgb_array, (w, h))

    def get_terminal_text(self):
        return self._cache_text

    def get_terminal_rgb_array(self):
        if self._cache_text_rgb_array is not None:
            return self._cache_text_rgb_array

        text = self._cache_text
        if text.strip() == "":
            return None
        text_img = text_to_rgb_array(text, self.font_name, self.font_size)
        if text_img is None:
            return None

        if self.scale != 1.0:
            import cv2

            w = int(text_img.shape[1] * self.scale)
            h = int(text_img.shape[0] * self.scale)
            text_img = cv2.resize(text_img, (w, h))

        self._cache_text_rgb_array = text_img
        return self._cache_text_rgb_array

    def get_rgb_array(self, return_terminal_image: bool = True):
        if self._cache_rgb_array is not None:
            return self._cache_rgb_array
        if return_terminal_image:
            return self.get_terminal_rgb_array()
        return None

    def render(self, **kwargs):
        """render関数の使用はあまり推奨せず、各項目を明示して実行が望ましい"""
        if self.base_render_target == "":
            pass
        elif self.base_render_target == "terminal":
            self.render_terminal()
        elif self.base_render_target == "terminal_to_text":
            return self._cache_text
        elif self.base_render_target == "terminal_to_rgb_array":
            return self._cache_text_rgb_array
        elif self.base_render_target == "rgb_array":
            return self._cache_rgb_array
        elif self.base_render_target == "window":
            self.render_window()
        else:
            raise UnimplementedCaseError()

    def render_terminal(self):
        print(self._cache_text, end="")

    def render_window(self):
        rgb_array = self._cache_rgb_array
        if rgb_array is None:
            rgb_array = self._cache_text_rgb_array
        if rgb_array is not None:
            self._render_window_sub(rgb_array, self.interval)
        return rgb_array

    def _render_window_sub(self, rgb_array, interval: float):
        import pygame

        from srl.utils import pygame_wrapper as pw

        if self._screen is None:
            if "SDL_VIDEODRIVER" in os.environ:
                pygame.display.quit()
                del os.environ["SDL_VIDEODRIVER"]

            pygame.init()
            w = min(rgb_array.shape[1], 1200)
            h = min(rgb_array.shape[0], 900)

            logger.info(f"create pygame({w},{h}), interval {interval}ms")
            self._screen = pygame.display.set_mode((w, h))
            self._t0 = time.time()

        pw.draw_image_rgb_array(self._screen, 0, 0, rgb_array)
        pygame.display.flip()

        # --- interval loop
        while True:
            pygame.event.get()

            if interval <= 0:
                break
            elapsed_time = time.time() - self._t0
            if elapsed_time > interval / 1000:
                break
        self._t0 = time.time()

        return rgb_array
