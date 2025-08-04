import importlib
import importlib.util
import logging
import os
import sys
from typing import Any, List, Optional, Type, Union, cast

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: Optional[int], enable_gpu: bool = False):
    if seed is None:
        return
    import random

    random.seed(seed)
    logger.debug(f"random.seed({seed})")
    np.random.seed(seed)
    logger.debug(f"np.random.seed({seed})")

    if is_package_imported("tensorflow"):
        import tensorflow as tf

        logger.debug(f"Tensorflow set_seed({seed})")
        tf.random.set_seed(seed)

        if enable_gpu:
            # GPU内の計算順の固定
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            logger.debug("set TF_DETERMINISTIC_OPS=1")

    if is_package_imported("torch"):
        import torch

        # https://pytorch.org/docs/stable/notes/randomness.html#reproducibility

        logger.debug(f"Torch set_seed({seed})")
        torch.manual_seed(seed)
        if enable_gpu:
            torch.cuda.manual_seed(seed)
            torch.use_deterministic_algorithms = True
            torch.backends.cudnn.benchmark = False  # type: ignore


def logger_print(
    level: Union[int, str] = logging.INFO,
    log_name: str = "",
    enable_log_extra_suppression: bool = True,
) -> None:
    set_logger(log_name, True, level, "", logging.DEBUG, enable_log_extra_suppression)


def logger_file(
    filename,
    level: Union[int, str] = logging.DEBUG,
    log_name: str = "",
    enable_log_extra_suppression: bool = True,
) -> None:
    set_logger(log_name, False, logging.INFO, filename, level, enable_log_extra_suppression)


def set_logger(
    log_name: str = "",
    print_terminal: bool = True,
    print_level: Union[int, str] = logging.INFO,
    filename: str = "",
    file_level: Union[int, str] = logging.DEBUG,
    enable_log_extra_suppression: bool = True,
) -> None:
    if isinstance(print_level, str):
        print_level = cast(int, logging.getLevelName(print_level.upper()))
    if isinstance(file_level, str):
        file_level = cast(int, logging.getLevelName(file_level.upper()))
    top_level = min(print_level, file_level)

    logger = logging.getLogger(log_name)
    logger.setLevel(top_level)
    fmt = "%(asctime)s %(name)s %(funcName)s %(lineno)d [%(levelname)s] %(message)s"
    formatter = logging.Formatter(fmt)

    if print_terminal:
        h = logging.StreamHandler()
        h.setLevel(print_level)
        h.setFormatter(formatter)
        logger.addHandler(h)

    if filename != "":
        h = logging.FileHandler(filename)
        h.setLevel(file_level)
        h.setFormatter(formatter)
        logger.addHandler(h)

    # root level
    root_logger = logging.getLogger()
    if root_logger.level < top_level:
        root_logger.setLevel(top_level)

    if enable_log_extra_suppression:
        import warnings

        # 余分なwarningを非表示
        warnings.simplefilter("ignore")

        # ライブラリ別にログレベルを調整
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.INFO)
        logging.getLogger("pika").setLevel(logging.ERROR)

        # TF log
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("h5py").setLevel(logging.INFO)


def save_file(path: str, dat: Any, compress: bool = True):
    import pickle

    try:
        if compress:
            import lzma

            dat = pickle.dumps(dat)
            with lzma.open(path, "w") as f:
                f.write(dat)
        else:
            with open(path, "wb") as f:
                pickle.dump(dat, f)
    except Exception:
        if os.path.isfile(path):
            os.remove(path)
        raise


def load_file(path: str) -> Any:
    import binascii
    import pickle

    # LZMA
    with open(path, "rb") as f:
        compress = binascii.hexlify(f.read(6)) == b"fd377a585a00"
    if compress:
        import lzma

        with lzma.open(path) as f:
            dat = f.read()
        dat = pickle.loads(dat)
    else:
        with open(path, "rb") as f:
            dat = pickle.load(f)
    return dat


def load_module(entry_point: str, partition: str = ":") -> Type[Any]:
    if "<locals>" in entry_point:
        raise ValueError(f"Cannot import local definitions: {entry_point}")
    module_path, _, attr_name = entry_point.rpartition(partition)
    if not module_path or not attr_name:
        raise ValueError(f"Invalid entry point format: {entry_point}")

    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def is_package_installed(name: str) -> bool:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return False
    return bool(spec)


def is_packages_installed(names: List[str]) -> bool:
    for name in names:
        if not is_package_installed(name):
            return False
    return True


def is_package_imported(name: str) -> bool:
    if name in sys.modules:
        if name == "tensorflow":
            import tensorflow as tf

            if tf.__file__ is None:  # なぜかuninstallしても残る
                return False
        return True
    return False


def is_env_notebook() -> bool:
    try:
        if get_ipython().__class__.__name__ == "TerminalInteractiveShell":  # type: ignore
            # IPython shell
            return False
    except NameError:
        # Python shell
        return False
    # Jupyter Notebook
    return True


def compare_less_package_version(name: str, v2) -> bool:
    module = importlib.import_module(name)
    v1 = module.__version__

    try:
        from packaging import version

        return version.parse(v1) < version.parse(v2)

    except ImportError:
        from distutils.version import LooseVersion

        return LooseVersion(v1) < LooseVersion(v2)


def compare_less_version(v1, v2) -> bool:
    try:
        from packaging import version

        return version.parse(v1) < version.parse(v2)

    except ImportError:
        from distutils.version import LooseVersion

        return LooseVersion(v1) < LooseVersion(v2)


def compare_equal_version(v1, v2) -> bool:
    try:
        from packaging import version

        return version.parse(v1) == version.parse(v2)

    except ImportError:
        from distutils.version import LooseVersion

        return LooseVersion(v1) == LooseVersion(v2)


def compare_between_version(v_low, v_target, v_high) -> bool:
    try:
        from packaging import version

        return version.parse(v_low) <= version.parse(v_target) < version.parse(v_high)

    except ImportError:
        from distutils.version import LooseVersion

        return LooseVersion(v_low) <= LooseVersion(v_target) < LooseVersion(v_high)


__cache_device_name = {}


def is_enable_tf_device_name(device_name) -> bool:
    global __cache_device_name

    if device_name == "":
        return False
    if device_name in __cache_device_name:
        return __cache_device_name[device_name]
    if not is_package_imported("tensorflow"):
        __cache_device_name[device_name] = False
        return False
    try:
        from tensorflow.python.client import device_lib
        from tensorflow.python.distribute import device_util

        full_device_name = device_util.canonicalize(device_name)

        for device in device_lib.list_local_devices():
            d = device_util.canonicalize(device.name)  # type: ignore
            if full_device_name == d:
                __cache_device_name[device_name] = True
                return True
    except (ValueError, ModuleNotFoundError):
        import traceback

        logger.info(traceback.format_exc())
    __cache_device_name[device_name] = False
    return False


def is_available_gpu_tf() -> bool:
    if not is_package_installed("tensorflow"):
        return False

    import tensorflow as tf

    if tf.__file__ is None:  # なぜかuninstallしても残る
        return False

    return len(tf.config.list_physical_devices("GPU")) > 0


def is_available_gpu_torch() -> bool:
    if not is_package_installed("torch"):
        return False

    import torch

    return torch.cuda.is_available()


def is_available_pygame_video_device() -> bool:
    if not is_package_installed("pygame"):
        return False

    import pygame

    SDL_VIDEODRIVER = os.environ.get("SDL_VIDEODRIVER", None)
    if "SDL_VIDEODRIVER" in os.environ:
        pygame.display.quit()
        del os.environ["SDL_VIDEODRIVER"]

    try:
        pygame.display.init()
        pygame.display.set_mode((1, 1))
        flag = True
    except pygame.error as e:
        logger.warning(f"pygame.error: {e}")
        flag = False
    finally:
        pygame.display.quit()
        if SDL_VIDEODRIVER is not None:
            os.environ["SDL_VIDEODRIVER"] = SDL_VIDEODRIVER

    return flag


def moving_average(data, rolling_size):
    if rolling_size <= 0:
        raise ValueError("rolling_size must be a positive integer")

    b = np.ones(rolling_size) / rolling_size
    y2 = np.convolve(data, b, mode="same")
    return y2


def ema(data, alpha: float = 0.1):
    """
    指数移動平均を計算する関数

    Parameters:
    data (numpy.ndarray): データの配列
    alpha (float): スムージング係数 (0 < alpha ≤ 1)

    Returns:
    numpy.ndarray: 指数移動平均の配列
    """
    ema_data = np.zeros_like(data)
    ema_data[0] = data[0]

    for i in range(1, len(data)):
        ema_data[i] = alpha * data[i] + (1 - alpha) * ema_data[i - 1]

    return ema_data


def line_smooth(
    data,
    window_size: int = 11,  # ウィンドウサイズ（奇数）
    poly_order: int = 2,  # 多項式の次数
):
    from scipy.signal import savgol_filter

    # サバン・ゴレイフィルタ
    if window_size % 2 == 0:
        window_size += 1
    return savgol_filter(data, window_size, poly_order)


def rolling(data, aggregation_num: int = 10):
    import pandas as pd

    if len(data) > aggregation_num * 1.2:
        window = int(len(data) / aggregation_num)
    else:
        window = 0

    if window > 1:
        return pd.Series(data).rolling(window).mean()
    else:
        return data
