import enum
import json
import logging
import traceback
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Dict, List, Literal, Union, cast, get_args, get_origin

import numpy as np

from srl.base.exception import NotSupportedError

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
    elif is_dataclass(data):
        data2 = [
            {k2: convert_for_json(v2) for k2, v2 in asdict(data).items()},
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


def update_dataclass_from_dict(obj: Any, data: Dict[str, Any]) -> Any:
    if not is_dataclass(obj):
        raise NotSupportedError("Provided object must be a dataclass instance.")

    for f in fields(obj):
        if f.name not in data:
            continue
        d = _update_dataclass_from_dict_sub(getattr(obj, f.name), cast(type, f.type), data[f.name])
        setattr(obj, f.name, d)


def _update_dataclass_from_dict_sub(obj_val: Any, type_: type, data: Any) -> Any:
    base_type = get_origin(type_)
    if base_type is None:
        base_type = type_
    if base_type is Union:
        # --- Union型はそれぞれの型で変換し、可能なものを採用する
        # Noneは先に見る
        for t in get_args(type_):
            if (t is type(None)) and (data is None):
                return None
        # 順番の優先順位は一旦保留、現状は先頭の型から優先
        d = None
        for t in get_args(type_):
            try:
                d = _update_dataclass_from_dict_sub(obj_val, t, data)
                break  # 変換できたらok
            except NotSupportedError:
                continue
        if d is None:
            raise NotSupportedError(f"'{data}' could not be converted to any of the Union types: {get_args(type_)}")
        return d
    elif base_type is None:
        return None
    elif base_type is Any:
        return data
    elif base_type in [int, float, bool, str]:
        return data
    elif base_type is bytes:
        # --- base64
        if isinstance(data, str):
            import base64

            try:
                return base64.b64decode(data)
            except Exception as e:
                logger.debug(traceback.format_exc())
                logger.info(e)
                return data.encode(errors="ignore")
        else:
            return data
    elif base_type is type:
        raise NotSupportedError(f"{base_type=}, {data=}")  # TODO?
    elif base_type is Literal:  # str
        return data
    elif base_type is list:
        # --- dataの方が多い場合、長さをそろえて上書きする
        # 短い場合はあるサイズまで上書きする
        if obj_val is None:
            obj_val = []
        for i in range(len(data)):
            if i >= len(obj_val):
                obj_val.append(None)
        tp = get_args(type_)
        tp = Any if len(tp) == 0 else tp[0]
        for i in range(len(obj_val)):
            if i >= len(data):
                break
            obj_val[i] = _update_dataclass_from_dict_sub(obj_val[i], cast(type, tp), data[i])
        return obj_val
    elif base_type is tuple:
        # --- 長さは見ずにそのまま上書きする
        return tuple(_update_dataclass_from_dict_sub(None, tp, v2) for tp, v2 in zip(get_args(type_), data))
    elif base_type is dict:
        # --- data側にあるkeyのみを上書きする
        if obj_val is None:
            obj_val = {}
        tp = get_args(type_)
        tp = Any if len(tp) == 0 else tp[1]
        for k in data.keys():
            if k in data:
                val = obj_val[k] if k in obj_val else None
                obj_val[k] = _update_dataclass_from_dict_sub(val, cast(Any, tp), data[k])
        return obj_val
    elif issubclass(base_type, enum.Enum):
        # --- strも変換
        if issubclass(type(data), enum.Enum):
            return data
        if data in base_type.__members__:
            return base_type[data]
        raise NotSupportedError(f"'{data}' is not a valid member of enum '{base_type.__name__}'")
    elif is_dataclass(base_type):
        # --- dataclass
        if obj_val is None:
            kwargs = {}
            for f in fields(base_type):
                if hasattr(data, f.name):
                    d = getattr(data, f.name)
                elif isinstance(data, dict) and (f.name in data):
                    d = data[f.name]
                else:
                    continue
                kwargs[f.name] = _update_dataclass_from_dict_sub(None, cast(type, f.type), d)
            return type_(**kwargs)
        else:
            for f in fields(base_type):
                if hasattr(data, f.name):
                    d = getattr(data, f.name)
                elif isinstance(data, dict) and (f.name in data):
                    d = data[f.name]
                else:
                    continue
                d = _update_dataclass_from_dict_sub(getattr(obj_val, f.name), cast(type, f.type), d)
                setattr(obj_val, f.name, d)
            return obj_val
    elif base_type is np.ndarray:
        # --- to_list <-> np.ndarray
        return np.asarray(data)
    elif isinstance(base_type, object):
        if isinstance(data, dict) and hasattr(base_type, "from_dict"):
            # --- to_dict/from_dict がある場合は実行する
            if obj_val is None:
                obj_val = base_type()  # type: ignore
            obj_val.from_dict(data)
            return obj_val
        else:
            return data
    elif callable(base_type):
        return data
    else:
        return data


def dataclass_to_dict(data: Any, exclude_names: List[str] = []) -> Any:
    """dataclassのパラメータをファイルで保存できる形式で返す
    - 変換できない型はそのまま返す
    """
    if data is None:
        return None
    elif type(data) in [int, float, bool, str]:
        return data
    elif isinstance(data, bytes):
        import base64

        return base64.b64encode(data).decode("utf-8")
    elif isinstance(data, list):
        return [dataclass_to_dict(d) for d in data]
    elif isinstance(data, tuple):
        return tuple(dataclass_to_dict(d) for d in data)
    elif isinstance(data, dict):
        return {
            k2: dataclass_to_dict(v2)  #
            for k2, v2 in data.items()
            if k2 not in exclude_names
        }
    elif issubclass(type(data), enum.Enum):
        return data.name
    elif is_dataclass(data):
        # dataclass -> dict
        return {
            f.name: dataclass_to_dict(getattr(data, f.name))  #
            for f in fields(data)
            if f.name not in exclude_names
        }
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, object):
        # to_dict/from_dict がある場合は実行する
        if hasattr(data, "to_dict"):
            return {
                k2: dataclass_to_dict(v2)  #
                for k2, v2 in data.to_dict().items()  # type: ignore , to_dict OK
                if k2 not in exclude_names
            }
        else:
            return data
    elif callable(data):
        return data
    else:
        return data
