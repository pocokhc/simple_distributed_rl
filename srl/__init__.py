import tensorflow

r""" 
gym -> tensorflow とimportすると以下の消えない警告がでるので先にimport

Python 3.9.13
gym                          0.24.1
tensorflow-gpu               2.9.1
Pillow                       9.1.1
flatbuffers                  1.12

\lib\site-packages\flatbuffers\compat.py:19: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
  import imp
\lib\site-packages\keras\utils\image_utils.py:36: DeprecationWarning: NEAREST is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.NEAREST or Dither.NONE instead.
  'nearest': pil_image.NEAREST,
\lib\site-packages\keras\utils\image_utils.py:37: DeprecationWarning: BILINEAR is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BILI  'bilinear': pil_image.BILINEAR,
\lib\site-packages\keras\utils\image_utils.py:38: DeprecationWarning: BICUBIC is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BICUBIC instead.
  'bicubic': pil_image.BICUBIC,
\lib\site-packages\keras\utils\image_utils.py:39: DeprecationWarning: HAMMING is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.HAMMING instead.
  'hamming': pil_image.HAMMING,
\lib\site-packages\keras\utils\image_utils.py:40: DeprecationWarning: BOX is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BOX instead.
  'box': pil_image.BOX,
\lib\site-packages\keras\utils\image_utils.py:41: DeprecationWarning: LANCZOS is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.LANCZOS instead.
  'lanczos': pil_image.LANCZOS,
"""

from . import envs, rl
from .version import VERSION as __version__  # noqa F402

__all__ = ["base", "envs", "rl", "runner", "test", "utils"]
