import dataclasses
import enum
import json
import logging
import traceback
from typing import Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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


def convert_for_json(data: Any) -> Any:
    """jsonがシリアライズ化できるように変換、復元は不可"""
    if data is None:
        data2 = None
    elif type(data) in [int, float, bool, str]:
        data2 = data
    elif isinstance(data, bytes):
        try:
            data2 = data.decode()
        except Exception:
            logger.info(traceback.format_exc())
            logger.warning(f"Decoding failed. Convert to string. {data}")
            data2 = str(data)
    elif type(data) in [list, tuple]:
        data2 = [convert_for_json(d) for d in data]
        if isinstance(data, tuple):
            data2 = tuple(data2)
    elif isinstance(data, dict):
        data2 = {k2: convert_for_json(v2) for k2, v2 in data.items()}
    elif issubclass(type(data), enum.Enum):
        data2 = [data.name, f"{data.__class__.__module__}.{data.__class__.__name__}"]
    elif dataclasses.is_dataclass(data):
        data2 = [
            {k2: convert_for_json(v2) for k2, v2 in dataclasses.asdict(data).items()},
            f"{data.__class__.__module__}.{data.__class__.__name__}",
        ]
    elif isinstance(data, np.ndarray):
        data2 = data.tolist()
    elif callable(data):
        data2 = f"{data.__module__}.{data.__name__}"
    elif isinstance(data, object):
        # to_dict がある場合は実行する
        if hasattr(data, "to_dict"):
            data2 = [
                {k2: convert_for_json(v2) for k2, v2 in data.to_dict().items()},  # type: ignore , to_dict OK
                f"{data.__class__.__module__}.{data.__class__.__name__}",
            ]
        else:
            data2 = f"{data.__class__.__module__}.{data.__class__.__name__}"
    else:
        data2 = str(data)

    return data2


def serialize_for_json(data: Any) -> Tuple[Any, Any]:
    """jsonがシリアライズ化できるように変換し復元も可能にしたい TODO"""
    if data is None:
        data2 = None
        base_type = "None"
    elif type(data) == int:
        data2 = data
        base_type = "int"
    elif type(data) == float:
        data2 = data
        base_type = "float"
    elif type(data) == bool:
        data2 = data
        base_type = "bool"
    elif type(data) == str:
        data2 = data
        base_type = "str"
    elif isinstance(data, bytes):
        try:
            data2 = data.decode()
            base_type = "bytes"
        except Exception:
            logger.info(traceback.format_exc())
            logger.warning(f"Decoding failed. Convert to string. {data}")
            data2 = str(data)
            base_type = "bytes_str"
    elif type(data) in [list, tuple]:
        data2 = []
        base_type = []
        for a in data:
            d2, t2 = serialize_for_json(a)
            data2.append(d2)
            base_type.append(t2)
        if isinstance(data, tuple):
            base_type = tuple(base_type)
    elif isinstance(data, dict):
        data2 = {}
        base_type = {}
        for k2, v2 in data.items():
            d2, t2 = serialize_for_json(v2)
            data2[k2] = d2
            base_type[k2] = t2
    elif issubclass(type(data), enum.Enum):
        data2 = [data.name, f"{data.__class__.__module__}.{data.__class__.__name__}"]
        base_type = "enum"
    elif dataclasses.is_dataclass(data):
        _d = {}
        for k2, v2 in dataclasses.asdict(data).items():
            d2, t2 = serialize_for_json(v2)
            _d[k2] = d2
        data2 = [_d, f"{data.__class__.__module__}.{data.__class__.__name__}"]
        base_type = "dataclass"
    elif isinstance(data, np.ndarray):
        data2 = data.tolist()
        base_type = "numpy"
    elif callable(data):
        data2 = f"{data.__module__}.{data.__name__}"
        base_type = "function"
    elif isinstance(data, object):
        # to_dict がある場合は実行する
        if hasattr(data, "to_dict"):
            data2 = [
                data.to_dict(),  # type: ignore , to_dict OK
                f"{data.__class__.__module__}.{data.__class__.__name__}",
            ]
            base_type = "class_dict"
        else:
            data2 = f"{data.__class__.__module__}.{data.__class__.__name__}"
            base_type = "class"
    else:
        data2 = str(data)
        base_type = ""

    return data2, base_type


def deserialize_for_json(data: Any, base_type: Any) -> Any:
    if base_type == "None":
        data2 = None
    elif base_type == "int":
        data2 = int(data)
    elif base_type == "float":
        data2 = float(data)
    elif base_type == "bool":
        data2 = bool(data)
    elif base_type == "str":
        data2 = str(data)
    elif base_type == "bytes":
        data2 = data.encode()
    elif base_type == "bytes_str":
        data2 = bytes(data)
    elif type(base_type) in [list, tuple]:
        data2 = []
        for d, t in zip(data, base_type):
            d2 = deserialize_for_json(d, t)
            data2.append(d2)
        if isinstance(base_type, tuple):
            data2 = tuple(data2)
    elif isinstance(data, dict):
        data2 = {}
        for k2, v2 in base_type.items():
            d2 = deserialize_for_json(data[k2], v2)
            data2[k2] = d2
    elif base_type == "enum":
        logger.info("Enums do not support deserialize.")  # TODO
        data2 = data[0]
        # data2 = importlib.import_module(data[1])[data[0]]
    elif base_type == "dataclass":
        logger.info("dataclass do not support deserialize.")  # TODO
        data2 = data[0]
        # data2 = importlib.import_module(data[1])(data[0])
    elif base_type == "numpy":
        data2 = np.array(data)
    elif base_type == "function":
        logger.info("function do not support deserialize.")  # TODO
        data2 = data[0]
    elif base_type == "class_dict":
        logger.info("class do not support deserialize.")  # TODO
        data2 = data[0]
        # data2 = importlib.import_module(data[1])()
        # if hasattr(data2, "from_dict"):
        #    data2.from_dict(data[0])
    elif base_type == "class":
        logger.info("class do not support deserialize.")  # TODO
        data2 = data
        # data2 = importlib.import_module(data)()
    else:
        data2 = data

    return data2
