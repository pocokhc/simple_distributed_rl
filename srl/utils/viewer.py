from typing import Optional, Tuple

import numpy as np

try:
    import pygame
except ImportError:
    pass


class Viewer:
    def __init__(self, width: int, height: int, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps

        pygame.init()
        pygame.display.init()
        self.clock = pygame.time.Clock()
        # self.screen = pygame.display.set_mode((self.width, self.height))
        self.screen = pygame.Surface((self.width, self.height))
        self.fonts = {}

        self.images = {}

    def __del__(self):
        self.close()

    def close(self):
        # pygame.display.quit()
        # pygame.quit()
        pass

    def registration_font(self, name: str, font: str = "arial", fontsize: int = 12):
        self.fonts[name] = pygame.font.SysFont(font, fontsize)

    def draw_start(self, color: Tuple[int, int, int] = (255, 255, 255)):
        self.screen.fill(color)

        # event
        # for event in pygame.event.get():
        #    pass

    def draw_end(self):
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.fps)

    def get_rgb_array(self) -> np.ndarray:
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    # --------------------
    def draw_fill(self, color: Tuple[int, int, int] = (255, 255, 255)):
        self.screen.fill(color)

    def draw_line(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int] = (0, 0, 0),
        width: int = 1,
    ):
        pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), width=width)

    def draw_box(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        filled: bool = True,
        fill_color: Tuple[int, int, int] = (255, 255, 255),
        width: int = 1,
        line_color: Tuple[int, int, int] = (0, 0, 0),
    ):
        if filled:
            self.screen.fill(fill_color, (x, y, w, h))

        if width > 0:
            lines = [
                (x, y),
                (x, y + h),
                (x + w, y + h),
                (x + w, y),
                (x, y),
            ]
            pygame.draw.lines(self.screen, line_color, False, lines, width=width)

    def draw_circle(
        self,
        x: int,
        y: int,
        radius: int,
        filled: bool = False,
        fill_color: Tuple[int, int, int] = (255, 255, 255),
        width: int = 1,
        line_color: Tuple[int, int, int] = (0, 0, 0),
    ):
        if filled:
            pygame.draw.circle(self.screen, fill_color, (x, y), radius, width=0)
        if width > 0:
            pygame.draw.circle(self.screen, line_color, (x, y), radius, width=width)

    def draw_text(
        self,
        x: int,
        y: int,
        text: str,
        fontname: str = "",
        color: Tuple[int, int, int] = (0, 0, 0),
    ):
        if fontname not in self.fonts:
            if fontname == "":
                self.fonts[""] = pygame.font.SysFont("arial", 12)
            else:
                raise ValueError()
        font = self.fonts[fontname]
        self.screen.blit(font.render(text, False, color), (x, y))

    def draw_image_rgb_array(
        self,
        x: int,
        y: int,
        rgb_array: np.ndarray,
        resize: Optional[Tuple[int, int]] = None,
    ):
        rgb_array = rgb_array.swapaxes(0, 1)
        img = pygame.surfarray.make_surface(rgb_array)
        if resize is not None:
            img = pygame.transform.scale(img, resize)
        self.screen.blit(img, (x, y))

    def load_image(self, name: str, path: str):
        self.images[name] = pygame.image.load(path)

    def draw_image(
        self,
        name: str,
        x: int,
        y: int,
        resize: Optional[Tuple[int, int]] = None,
    ):
        img = self.images[name]
        self.screen.blit(img, (x, y))
