import io
import logging
import sys
from typing import Callable

import numpy as np

from srl.font import get_font_path
from srl.utils.common import compare_less_version

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

    global _g_fonts

    if text == "":
        return np.zeros((1, 1, 3), dtype=np.uint8)

    if font_name == "":
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
