import io
import logging
import sys
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def print_to_text(print_function: Callable[[], None]) -> str:
    text = ""
    _stdout = sys.stdout
    try:
        sys.stdout = io.StringIO()
        print_function()
        text = sys.stdout.getvalue()
    except NotImplementedError:
        pass
    finally:
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.stdout = _stdout
    return text


_g_fonts = {}


def text_to_rgb_array(
    text: str,
    font_name: str = "",
    font_size: int = 18,
) -> np.ndarray:
    import PIL.Image
    import PIL.ImageDraw
    import PIL.ImageFont

    from srl.utils.common import compare_less_version

    global _g_fonts

    if text == "":
        return np.zeros((1, 1, 3), dtype=np.uint8)

    if font_name == "":
        from srl.font import get_font_path

        font_name = get_font_path()
    if font_name in _g_fonts:
        font = _g_fonts[font_name]
    else:
        logger.info(f"load font: {font_name}")
        font = PIL.ImageFont.truetype(font_name, size=font_size)
        _g_fonts[font_name] = font

    texts = text.split("\n")
    texts = texts[:49] + ["..."] if len(texts) >= 50 else texts
    texts = [t[:197] + "..." if len(t) > 200 else t for t in texts]
    text = "\n".join(texts)

    canvas_size = (640, 480)
    img = PIL.Image.new("RGB", canvas_size)
    draw = PIL.ImageDraw.Draw(img)
    if compare_less_version(PIL.__version__, "9.2.0"):
        text_width, text_height = draw.multiline_textsize(text, font=font)
    else:
        _, _, text_width, text_height = draw.multiline_textbbox((0, 0), text, font=font)

    canvas_size = (text_width, text_height)
    background_rgb = (0, 0, 0)
    text_rgb = (255, 255, 255)
    img = PIL.Image.new("RGB", canvas_size, background_rgb)
    draw = PIL.ImageDraw.Draw(img)
    draw.text((0, 0), text, fill=text_rgb, font=font)
    img = np.array(img).astype(np.uint8)

    return img


def draw_text(
    img: np.ndarray,
    x: int,
    y: int,
    text: str,
    fontScale: float = 0.5,
    color=(255, 255, 255),
    thickness=1,
    outline: bool = True,
    outline_color=(0, 0, 0),
):
    import cv2

    fontFace = cv2.FONT_HERSHEY_SIMPLEX

    if outline:
        img = cv2.putText(
            img,
            text,
            (x, y),
            fontFace,
            fontScale,
            color=outline_color,
            thickness=thickness + 2,
        )
    img = cv2.putText(
        img,
        text,
        (x, y),
        fontFace,
        fontScale,
        color=color,
        thickness=thickness,
    )
    return img


def add_border(image: np.ndarray, border_width: int, border_color=(255, 255, 255)):
    height, width = image.shape[:2]

    new_image = np.zeros((height + border_width * 2, width + border_width * 2, 3), dtype=np.uint8)
    new_image[border_width : height + border_width, border_width : width + border_width] = image

    new_image[:border_width, :] = border_color
    new_image[-border_width:, :] = border_color
    new_image[:, :border_width] = border_color
    new_image[:, -border_width:] = border_color
    return new_image


def add_padding(img: np.ndarray, top: int, bottom: int, left: int, right: int, color=(0, 0, 0)):
    import cv2

    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def vconcat(img1: np.ndarray, img2: np.ndarray, color1=(0, 0, 0), color2=(0, 0, 0)):
    # 垂直
    import cv2

    maxw = max(img1.shape[1], img2.shape[1])
    img1 = cv2.copyMakeBorder(img1, 0, 0, 0, maxw - img1.shape[1], cv2.BORDER_CONSTANT, value=color1)
    img2 = cv2.copyMakeBorder(img2, 0, 0, 0, maxw - img2.shape[1], cv2.BORDER_CONSTANT, value=color2)
    return cv2.vconcat([img1, img2])


def hconcat(img1: np.ndarray, img2: np.ndarray, color1=(0, 0, 0), color2=(0, 0, 0)):
    # 水平
    import cv2

    maxh = max(img1.shape[0], img2.shape[0])
    img1 = cv2.copyMakeBorder(img1, 0, maxh - img1.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=color1)
    img2 = cv2.copyMakeBorder(img2, 0, maxh - img2.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=color2)
    return cv2.hconcat([img1, img2])
