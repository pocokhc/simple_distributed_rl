import glob
import importlib
import importlib.util
import logging
import os
import sys
from types import ModuleType

logger = logging.getLogger(__name__)


def load_native_module(module_name: str) -> ModuleType:
    """OSに応じたバイナリモジュール（.so / .pyd）をロードする"""

    ext = {
        "win32": ".pyd",
        "linux": ".so",
        "darwin": ".so",  # Why not: macOS も .so 形式
    }.get(sys.platform, None)
    if ext is None:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")

    base_dir = os.path.join(os.path.dirname(__file__), sys.platform)

    # モジュール候補を探索
    pattern = os.path.join(base_dir, f"{module_name}*{ext}")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No native module found for pattern: {pattern}")
    logger.info(f"Found native module candidates: {candidates}")

    # 最初に見つかった候補をロード
    module_path = candidates[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for {module_name} from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
