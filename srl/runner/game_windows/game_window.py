import enum
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import pygame

from srl.utils import pygame_wrapper as pw
from srl.utils.render_functions import add_border

logger = logging.getLogger(__name__)


class KeyStatus(enum.Enum):
    UP = enum.auto()  # 離している間
    DOWN = enum.auto()  # 押している間
    PRESSED = enum.auto()  # 押した瞬間のみ
    RELEASED = enum.auto()  # 離した瞬間のみ


class GameWindow(ABC):
    def __init__(self, _is_test: bool = False) -> None:
        self._is_test = _is_test  # for test

        self.title: str = ""
        self.padding: int = 4
        self.img_dir = os.path.join(os.path.dirname(__file__), "img")
        self.keys_status = {}
        self._valid_unicode_keys = ["-", "+"]

        self.org_env_w = 0
        self.org_env_h = 0
        self.rl_w = 0
        self.rl_h = 0
        self.rl_state_w = 0
        self.rl_state_h = 0
        self.info_w = 100
        self.info_h = 100
        self.scale = 1
        self.resize(1.0)

    @abstractmethod
    def on_loop(self, events: List[pygame.event.Event]):
        raise NotImplementedError()

    def get_key(self, key) -> KeyStatus:
        return self.keys_status.get(key, KeyStatus.UP)

    def get_down_keys(self) -> List[int]:
        keys = [k for k, s in self.keys_status.items() if s == KeyStatus.DOWN]
        return [k for k in keys if k not in self._valid_unicode_keys]

    def get_pressed_keys(self) -> List[int]:
        keys = [k for k, s in self.keys_status.items() if s == KeyStatus.PRESSED]
        return [k for k in keys if k not in self._valid_unicode_keys]

    def play(self):
        if "SDL_VIDEODRIVER" in os.environ:
            pygame.display.quit()
            del os.environ["SDL_VIDEODRIVER"]

        # --- pygame init
        self.base_info_x = self.env_w + self.padding
        self.base_info_y = self.padding

        result = pygame.init()
        logger.info(f"pygame init: {result}")
        pygame.display.set_caption(self.title)
        pygame.display.set_mode((900, 600))
        self.resize(1.0)
        clock = pygame.time.Clock()
        pygame.key.set_repeat(500, 30)

        # -------------------------------
        # pygame window loop
        # -------------------------------
        self.pygame_done = False
        while not self.pygame_done:
            # --- key check
            for k in self.keys_status.keys():
                if self.keys_status[k] == KeyStatus.RELEASED:
                    self.keys_status[k] = KeyStatus.UP
                if self.keys_status[k] == KeyStatus.PRESSED:
                    self.keys_status[k] = KeyStatus.DOWN

            # --- event check
            is_window_resize = False
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.pygame_done = True
                elif event.type == pygame.KEYUP:
                    self.keys_status[event.key] = KeyStatus.RELEASED
                    if event.unicode in self._valid_unicode_keys:
                        self.keys_status[event.unicode] = KeyStatus.RELEASED

                elif event.type == pygame.KEYDOWN:
                    if self.keys_status.get(event.key, KeyStatus.UP) != KeyStatus.DOWN:
                        self.keys_status[event.key] = KeyStatus.PRESSED
                    if event.unicode in self._valid_unicode_keys:
                        if self.keys_status.get(event.unicode, KeyStatus.UP) != KeyStatus.DOWN:
                            self.keys_status[event.unicode] = KeyStatus.PRESSED

                    if event.key == pygame.K_ESCAPE:
                        self.pygame_done = True
                    elif event.unicode == "1":
                        self.scale = 0.5
                        is_window_resize = True
                    elif event.unicode == "2":
                        self.scale = 1.0
                        is_window_resize = True
                    elif event.unicode == "3":
                        self.scale = 1.5
                        is_window_resize = True
                    elif event.unicode == "4":
                        self.scale = 2.0
                        is_window_resize = True
                    elif event.unicode == "5":
                        self.scale = 3.0
                        is_window_resize = True
                    elif event.unicode == "6":
                        self.scale = 4.0
                        is_window_resize = True

            # --- window check
            if self.org_env_w < self.env_image.shape[1]:
                self.org_env_w = self.env_image.shape[1]
                is_window_resize = True
            if self.org_env_h < self.env_image.shape[0]:
                self.org_env_h = self.env_image.shape[0]
                is_window_resize = True
            if self.rl_w < self.rl_image.shape[1]:
                self.rl_w = self.rl_image.shape[1]
                is_window_resize = True
            if self.rl_h < self.rl_image.shape[0]:
                self.rl_h = self.rl_image.shape[0]
                is_window_resize = True
            if self.rl_state_w < self.rl_state_image.shape[1]:
                self.rl_state_w = self.rl_state_image.shape[1]
                is_window_resize = True
            if self.rl_state_h < self.rl_state_image.shape[0]:
                self.rl_state_h = self.rl_state_image.shape[0]
                is_window_resize = True

            self.screen.fill((0, 0, 0))

            # --- image
            pw.draw_image_rgb_array(self.screen, self.base_rl_x, self.base_rl_y, self.rl_image)
            pw.draw_image_rgb_array(self.screen, 0, self.env_h, self.rl_state_image)
            pw.draw_image_rgb_array(
                self.screen,
                0,
                0,
                self.env_image,
                resize=(
                    int(self.env_image.shape[1] * self.scale),
                    int(self.env_image.shape[0] * self.scale),
                ),
            )

            # --- info
            self.info_texts = []
            self.hotkey_texts = [
                "- Hotkeys -",
                f"1-6: change screen size (x{self.scale:.1f})",
            ]

            self.on_loop(events)
            self.info_texts.append("")
            self.info_texts.extend(self.hotkey_texts)
            width, height = pw.draw_texts(
                self.screen,
                self.base_info_x,
                self.base_info_y,
                self.info_texts,
                size=16,
                color=(255, 255, 255),
            )
            if self.info_w < width:
                self.info_w = width
                is_window_resize = True
            if self.info_h < height:
                self.info_h = height
                is_window_resize = True

            pygame.display.flip()
            clock.tick(60)

            if is_window_resize:
                self.resize(self.scale)

            if self._is_test:
                self.pygame_done = True

    def set_image(self, env_image: np.ndarray, rl_image: Optional[np.ndarray], rl_state_image: Optional[np.ndarray]):
        self.env_image = add_border(env_image, 1)
        self.rl_image = np.zeros((0, 0, 3), np.uint8) if rl_image is None else rl_image

        if rl_state_image is not None:
            self.rl_state_image = add_border(rl_state_image, 1)
        else:
            self.rl_state_image = np.zeros((0, 0, 3), np.uint8)

    def add_hotkey_texts(self, texts: List[str]):
        self.hotkey_texts.extend(texts)

    def add_info_texts(self, texts: List[str]):
        self.info_texts.extend(texts)

    def resize(self, scale: float):
        self.scale = scale
        self.env_w = self.org_env_w * scale
        self.env_h = self.org_env_h * scale

        self.base_rl_x = self.env_w + self.padding
        self.base_rl_y = self.padding
        self.base_info_x = self.base_rl_x + self.rl_w + self.padding
        self.base_info_y = self.padding

        window_w = self.env_w + self.padding + self.rl_w + self.padding + self.info_w
        window_h = max(max(self.env_h + self.rl_state_h, self.rl_h), self.info_h) + self.padding * 2

        window_w = min(window_w, 1400)
        window_h = min(window_h, 1000)
        self.screen = pygame.display.set_mode((window_w, window_h))

    def draw_texts(
        self,
        x: float,
        y: float,
        texts: List[str],
        size: int = 16,
        color: Tuple[int, int, int] = (255, 255, 255),
    ):
        pw.draw_texts(self.screen, x, y, texts, color=color, size=size)
