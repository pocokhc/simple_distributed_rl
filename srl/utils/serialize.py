import dataclasses
import enum
import json
import logging
import traceback
from typing import Any

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
        data2 = data.name
        # data2 = [data.name, f"{data.__class__.__module__}.{data.__class__.__name__}"]
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
