import importlib
import json
import logging
import os
import random
import sys
import traceback
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: Optional[int], enable_gpu: bool = False):
    if seed is None:
        return
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

    if is_package_imported("torch"):
        import torch

        # https://pytorch.org/docs/stable/notes/randomness.html#reproducibility

        logger.debug(f"Torch set_seed({seed})")
        torch.manual_seed(seed)
        if enable_gpu:
            torch.cuda.manual_seed(seed)
            torch.use_deterministic_algorithms = True
            torch.backends.cudnn.benchmark = False


def logger_print(
    level: Union[int, str] = logging.INFO,
    log_name: str = "",
    enable_log_extra_suppression: bool = True,
) -> None:
    set_logger(log_name, True, level, "", "DEBUG", enable_log_extra_suppression)


def logger_file(
    filename,
    level: Union[int, str] = logging.DEBUG,
    log_name: str = "",
    enable_log_extra_suppression: bool = True,
) -> None:
    set_logger(log_name, False, "INFO", filename, level, enable_log_extra_suppression)


def set_logger(
    log_name: str = "",
    print_terminal: bool = True,
    print_level: Union[int, str] = logging.INFO,
    filename: str = "",
    file_level: Union[int, str] = logging.DEBUG,
    enable_log_extra_suppression: bool = True,
) -> None:
    if isinstance(print_level, str):
        print_level: int = logging.getLevelName(print_level)
    if isinstance(file_level, str):
        file_level: int = logging.getLevelName(file_level)
    top_level = print_level if (print_level > file_level) else file_level

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
        # 余分なwarningを非表示
        warnings.simplefilter("ignore")

        # ライブラリ別にログレベルを調整
        logging.getLogger("matplotlib").setLevel(logging.INFO)
        logging.getLogger("PIL").setLevel(logging.INFO)

        # TF log
        # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("h5py").setLevel(logging.INFO)


def listdict_to_dictlist(data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    _info = {}
    for h in data:
        if h is None:
            continue
        for key, val in h.items():
            if key not in _info:
                _info[key] = []
            _info[key].append(val)
    return _info


def listdictdict_to_dictlist(data: List[Dict[str, Dict[str, Any]]], key: str) -> Dict[str, List[Any]]:
    _info = {}
    for h in data:
        if key not in h:
            continue
        if h[key] is None:
            continue
        for k, val in h[key].items():
            if val is None:
                continue
            if k not in _info:
                _info[k] = []
            _info[k].append(val)
    return _info


def summarize_info_from_dictlist(info: Dict[str, List[Any]]) -> Dict[str, Union[float, str]]:
    new_info = {}
    for k, arr in info.items():
        new_info[k] = summarize_info_from_list(arr)
    return new_info


def summarize_info_from_list(arr: List[Any]) -> Union[float, str, None]:
    # 数字が入っていればそちらを優先、文字が入っていれば最後を反映
    vals = []
    last_str = ""
    for v in arr:
        if v is None:
            continue
        if isinstance(v, int):
            vals.append(v)
        elif isinstance(v, float):
            vals.append(v)
        elif isinstance(v, np.integer):
            vals.append(v)
        elif isinstance(v, np.floating):
            vals.append(v)
        elif isinstance(v, np.ndarray):
            vals.append(np.mean(v))
        else:
            last_str = v
    if len(vals) == 0:
        last_str = str(last_str)
        if last_str == "":
            return None
        else:
            return last_str
    else:
        return float(np.mean(vals))


def to_str_time(sec: float) -> str:
    if sec == np.inf:
        return "   inf"
    if sec < 60:
        return "{:5.2f}s".format(sec)
    return "{:5.1f}m".format(sec / 60)


class JsonNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonNumpyEncoder, self).default(obj)


def load_module(entry_point: str):
    if ":" not in entry_point:
        raise ValueError(f"entry_point must include ':'({entry_point})")

    mod_name, cls_name = entry_point.split(":")
    module = importlib.import_module(mod_name)
    cls = getattr(module, cls_name)
    return cls


_package_cache = {}


def is_package_installed(name: str) -> bool:
    global _package_cache

    if name in _package_cache:
        return _package_cache[name]

    try:
        importlib.import_module(name)
        result = True
    except ImportError:
        result = False

    _package_cache[name] = result
    return result


def is_packages_installed(names: List[str]) -> bool:
    for name in names:
        if not is_package_installed(name):
            return False
    return True


def is_package_imported(name: str) -> bool:
    return name in sys.modules


def is_env_notebook() -> bool:
    try:
        if get_ipython().__class__.__name__ == "TerminalInteractiveShell":
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


def is_enable_tf_device_name(device_name) -> bool:
    if device_name == "":
        return False
    if not is_package_imported("tensorflow"):
        return False
    try:
        from tensorflow.python.client import device_lib
        from tensorflow.python.distribute import device_util

        full_device_name = device_util.canonicalize(device_name)

        for device in device_lib.list_local_devices():
            d = device_util.canonicalize(device.name)
            if full_device_name == d:
                return True
    except ValueError:
        logger.info(traceback.format_exc())
    return False


def is_available_gpu_tf():
    import tensorflow as tf

    return len(tf.config.list_physical_devices("GPU")) > 0


def is_available_gpu_torch():
    import torch

    return torch.cuda.is_available()
