import glob
import importlib
import importlib.util
import logging
import os
import sys
from types import ModuleType
from typing import Optional

logger = logging.getLogger(__name__)


def _find_module_path(module_name: str) -> Optional[str]:
    ext = {
        "win32": ".pyd",
        "linux": ".so",
        "darwin": ".so",  # Why not: macOS も .so 形式
    }.get(sys.platform, None)
    if ext is None:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")

    base_dir = os.path.join(os.path.dirname(__file__), "bin", sys.platform)

    # モジュール候補を探索
    pattern = os.path.join(base_dir, f"{module_name}*{ext}")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return candidates[0]


__g_loaded_force_build = False


def load_or_build_module(module_name: str, force_build: bool = False) -> ModuleType:
    module_name_cpp = f"{module_name}_cpp"

    # 既にloadされていたらskip
    if module_name_cpp in sys.modules:
        return sys.modules[module_name_cpp]

    module_path = _find_module_path(module_name_cpp)
    if (module_path is not None) and force_build:
        global __g_loaded_force_build

        if os.path.isfile(module_path) and (not __g_loaded_force_build):
            os.remove(module_path)
            module_path = None
            __g_loaded_force_build = True

    if module_path is None:
        # ビルドしてからロードする、以下の理由により subprocess 経由で実行する
        # setup() is intended to be called once per Python process.
        import subprocess

        try:
            print("--- start build module")
            subprocess.run(
                [sys.executable, "setup_cpp.py", module_name],
                cwd=os.path.dirname(__file__),
                check=True,  # エラー発生時に例外を投げる
            )
            print("--- end build module")
        except Exception:
            s = (
                "\n[ERROR] Failed to build the C++ extension module required for `set_proportional_memory_cpp()`.\n"
                "This error typically occurs when your environment is not set up for C++ compilation.\n"
                "\nTo fix this, ensure the following are installed:\n"
                "  - On Linux: `g++` (e.g. `sudo apt install g++ build-essential`)\n"
                "  - On macOS: Xcode Command Line Tools (e.g. `xcode-select --install`)\n"
                "  - On Windows: Visual Studio Build Tools with C++ CMake and MSVC components\n"
                "\nAlso, make sure `pybind11` `setuptools` is installed:\n"
                "  pip install pybind11 setuptools\n"
                "\nIf you cannot build the module or do not need high-performance C++ code,\n"
                "consider using `set_proportional_memory()` instead, which is implemented purely in Python.\n"
            )
            print(s)
            logger.info(s)
            raise

        module_path = _find_module_path(module_name_cpp)
        if module_path is None:
            raise RuntimeError(f"Failed to find the module after building: {module_name}")

    # module load
    spec = importlib.util.spec_from_file_location(module_name_cpp, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for {module_name_cpp} from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
