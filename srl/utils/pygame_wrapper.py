import os
from typing import List, Optional, Tuple

import numpy as np
import pygame
from srl.font import get_font_path


def create_surface(width: int, height: int) -> pygame.surface.Surface:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    pygame.display.init()
    return pygame.Surface((width, height))


def get_rgb_array(surface: pygame.surface.Surface) -> np.ndarray:
    return np.transpose(np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2))


def draw_fill(surface: pygame.surface.Surface, color: Tuple[int, int, int] = (255, 255, 255)):
    surface.fill(color)


def draw_line(
    surface: pygame.surface.Surface,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: Tuple[int, int, int] = (0, 0, 0),
    width: int = 1,
):
    pygame.draw.line(
        surface,
        color,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        width=width,
    )


def draw_box(
    surface: pygame.surface.Surface,
    x: float,
    y: float,
    w: float,
    h: float,
    filled: bool = True,
    fill_color: Tuple[int, int, int] = (255, 255, 255),
    width: int = 1,
    line_color: Tuple[int, int, int] = (0, 0, 0),
):
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    if filled:
        surface.fill(fill_color, (x, y, w, h))

    if width > 0:
        lines = [
            (x, y),
            (x, y + h),
            (x + w, y + h),
            (x + w, y),
            (x, y),
        ]
        pygame.draw.lines(surface, line_color, False, lines, width=width)


def draw_circle(
    surface: pygame.surface.Surface,
    x: float,
    y: float,
    radius: float,
    filled: bool = False,
    fill_color: Tuple[int, int, int] = (255, 255, 255),
    width: float = 1,
    line_color: Tuple[int, int, int] = (0, 0, 0),
):
    x = int(x)
    y = int(y)
    radius = int(radius)
    width = int(width)
    if filled:
        pygame.draw.circle(surface, fill_color, (x, y), radius, width=0)
    if width > 0:
        pygame.draw.circle(surface, line_color, (x, y), radius, width=width)


_fonts = {}


def draw_text(
    surface: pygame.surface.Surface,
    x: float,
    y: float,
    text: str,
    font: str = "",
    size: int = 12,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[int, int]:  # width, height
    global _fonts

    x = int(x)
    y = int(y)

    font_key = f"{font}_{size}"
    if font_key not in _fonts:
        if font == "":
            pygame.font.init()
            font = get_font_path()
        _fonts[font_key] = pygame.font.Font(font, size)

    _font = _fonts[font_key]
    surface.blit(_font.render(text, False, color), (x, y))
    return _font.size(text)


def draw_texts(
    surface: pygame.surface.Surface,
    x: float,
    y: float,
    texts: List[str],
    font: str = "",
    size: int = 12,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[int, int]:  # width, height:
    height = 0
    width = 0
    for i, text in enumerate(texts):
        w, h = draw_text(surface, x, y + height, text, font, size, color)
        width = max(width, w)
        height += h
    return width, height


def draw_image_rgb_array(
    surface: pygame.surface.Surface,
    x: float,
    y: float,
    rgb_array: np.ndarray,
    resize: Optional[Tuple[int, int]] = None,
):
    x = int(x)
    y = int(y)
    rgb_array = rgb_array.swapaxes(0, 1)
    img = pygame.surfarray.make_surface(rgb_array)
    if resize is not None:
        img = pygame.transform.scale(img, resize)
    surface.blit(img, (x, y))


_images = {}


def load_image(name: str, path: str):
    global _images
    _images[name] = pygame.image.load(path)


def draw_image(
    surface: pygame.surface.Surface,
    name: str,
    x: float,
    y: float,
    resize: Optional[Tuple[int, int]] = None,
):
    global _images

    x = int(x)
    y = int(y)
    img = _images[name]
    if resize is not None:
        img = pygame.transform.scale(img, resize)
    surface.blit(img, (x, y))
