import importlib
import json
import logging
import pkgutil
import warnings
from typing import Any, Dict, List

import numpy as np


def set_logger(
    name: str = "",
    print_terminal: bool = True,
    print_level=logging.INFO,
    filename: str = "",
    file_level=logging.DEBUG,
) -> None:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(funcName)s %(lineno)d [%(levelname)s] %(message)s")

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


def to_str_time(sec: float) -> str:
    if sec == np.inf:
        return "    inf"
    if sec < 60:
        return "{:6.2f}s".format(sec)
    return "{:6.1f}m".format(sec / 60)


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
    except ImportError:
        pass

    for m in pkgutil.iter_modules():
        if m.name == name:
            _package_cache[name] = True
            return True

    _package_cache[name] = False
    return False


def is_env_notebook():
    try:
        if get_ipython().__class__.__name__ == "TerminalInteractiveShell":
            # IPython shell
            return False
    except NameError:
        # Python shell
        return False
    # Jupyter Notebook
    return True
