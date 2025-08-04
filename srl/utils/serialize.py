import enum
import inspect
import json
import logging
import os
import pickle
import traceback
from dataclasses import fields, is_dataclass
from types import FunctionType
from typing import Any, Callable, List, Literal, Sequence, Tuple, Union, cast, get_args, get_origin

import numpy as np

from srl.base.exception import NotSupportedError
from srl.utils.common import load_module

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


class JsonSafeEncoder(json.JSONEncoder):
    """
    JSONにシリアライズできないオブジェクトはNoneに変換するエンコーダー

    How:
    json.dumps(obj, cls=JsonSafeEncoder) のように使う
    """

    def default(self, obj: Any) -> Any:
        try:
            return super().default(obj)
        except (TypeError, OverflowError):
            return None


def save_dict(dat: dict, path: str):
    """
    dictのインスタンスを指定されたファイルパスに保存する関数

    - ファイル形式は拡張子によって自動判定する（json, yaml対応）
    - 保存時のエンコーディングはUTF-8固定

    Parameters
    ----------
    dat : dict
        保存対象のdictインスタンス
    path : str
        保存先のファイルパス（拡張子で形式を判別）
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in {".yaml", ".yml"}:
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dat, f, allow_unicode=True, sort_keys=False)
    else:
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(dat, f, indent=2, ensure_ascii=False, cls=JsonSafeEncoder)


def load_dict(path_or_cfg_dict: Union[dict, Any, str]) -> dict:
    """
    設定ファイルまたは辞書から値を読み込む

    - path_or_cfg_dict が文字列の場合は、そのパスからファイルを読み込む

    Parameters
    ----------
    path_or_cfg_dict : dict | str | Any
        - dict: 上書きするキーと値の辞書
        - str: JSON/YAML/TOML形式のファイルパス
        - Any: 辞書と同等のマッピング可能なオブジェクト
    """

    if isinstance(path_or_cfg_dict, str):
        path = path_or_cfg_dict
        ext = os.path.splitext(path)[1].lower()
        if ext in {".yaml", ".yml"}:
            import yaml

            with open(path, "r", encoding="utf-8") as f:
                dat = yaml.safe_load(f)
        else:
            import json

            with open(path, "r", encoding="utf-8") as f:
                dat = json.load(f)
    else:
        dat = path_or_cfg_dict

    return dat


def apply_dict_to_dataclass(obj: Any, cfg_dict: Union[dict, Any], strict: bool = False) -> Any:
    """
    辞書から与えられた値で dataclass インスタンスを再帰的に更新する

    - `cfg_dict` に存在するキーに対応するフィールドが `obj` に存在する場合、その値を更新する
    - ネストされた dataclass や list、dict などの再帰構造にも対応
    - 存在しないキーは無視される
    - 非dataclass型や変換できない型の場合は例外が発生する可能性がある

    Args:
        obj (Any): 更新対象の dataclass インスタンス
        cfg_dict (Any): 更新に使用するキーと値の辞書

    Returns:
        Any: 更新後の dataclass インスタンス（元のオブジェクトと同じインスタンス）

    Raises:
        TypeError: obj が dataclass インスタンスでない場合
        ValueError: 値の変換に失敗した場合
    """
    return _apply_dict_to_dataclass_sub(type(obj), "", obj, cfg_dict, strict, depth=0)


def _apply_dict_to_dataclass_sub(type_: type, key: str, src_data: Any, data: Union[dict, Any], strict: bool, depth: int) -> Any:
    logger.debug("  " * depth + f"{type_} key='{key}' src={str(src_data)[:20]}..., dest={str(data)[:20]}...")

    # 文字列
    if isinstance(type_, str):
        type_ = Any  # type: ignore

    # --- check Optional, Union
    base_type = get_origin(type_)
    if base_type is None:
        base_type = type_

    # literalは文字列にする
    if base_type is Literal:
        base_type = str

    # Anyはdataを採用
    if base_type is Any:
        base_type = type(data)

    # --- Union型はそれぞれの型で変換し、可能なものを採用する
    if base_type is Union:
        # Noneは先に見る
        for t in get_args(type_):
            if (t is type(None)) and (data is None):
                return None
        # 順番の優先順位は一旦保留、現状は先頭の型から優先
        d = None
        for t in get_args(type_):
            try:
                d = _apply_dict_to_dataclass_sub(t, key, src_data, data, strict=True, depth=depth)
                break  # 変換できたらok
            except NotSupportedError:
                continue
            except TypeError:
                continue
        if d is None:
            raise NotSupportedError(f"'{data}' could not be converted to any of the Union types: {get_args(type_)}")
        return d

    if not isinstance(base_type, type):
        base_type = type(base_type)

    # --- 特殊処理、data側にtargetがある場合はクラスにする
    def _is_dict(d):
        if isinstance(d, dict) and ("_target_" in d):
            return True
        if inspect.isclass(type(d)) and type(d).__module__ != "builtins":
            if getattr(d, "__getitem__", None):
                if "_target_" in d:
                    return True
        return False

    if _is_dict(data):
        data = cast(dict, data)
        attr = load_module(data["_target_"], partition=".")

        # 関数
        if inspect.isfunction(attr):
            if strict and (not (isinstance(base_type, FunctionType) or issubclass(base_type, Callable))):
                raise TypeError()
            return attr

        # クラス
        if depth > 0 and strict:
            if base_type in (int, float, bool, str, bytes, complex, type(None)):
                raise TypeError()
        args = []
        if "_args_" in data:
            if strict and (not isinstance(data["_args_"], list)):
                raise TypeError()
            args = _apply_dict_to_dataclass_sub(list, "", None, data["_args_"], strict, depth + 1)

        if is_dataclass(attr):
            if src_data is None:
                kwargs = {}
                for f in fields(attr):
                    if f.name in data:
                        d = data[f.name]
                    else:
                        continue
                    kwargs[f.name] = _apply_dict_to_dataclass_sub(cast(type, f.type), f.name, None, d, strict, depth + 1)
                if strict:
                    try:
                        return attr(*args, **kwargs)
                    except Exception:
                        raise TypeError()
                else:
                    return attr(*args, **kwargs)
            else:
                for f in fields(attr):
                    if f.name in data:
                        d = data[f.name]
                    else:
                        continue
                    src = getattr(src_data, f.name) if hasattr(src_data, f.name) else None
                    d = _apply_dict_to_dataclass_sub(cast(type, f.type), f.name, src, d, strict, depth + 1)
                    setattr(src_data, f.name, d)
                return src_data
        else:
            if src_data is None:
                if strict:
                    try:
                        src_data = attr(*args)
                    except Exception:
                        raise TypeError()
                else:
                    src_data = attr(*args)
            for k, v in data.items():
                if k in ["_target_", "_args_"]:
                    continue
                if not hasattr(src_data, k):
                    continue
                v = _apply_dict_to_dataclass_sub(type(v), key, getattr(src_data, k), v, strict, depth + 1)
                setattr(src_data, k, v)
            return src_data

    # --- 各type
    if (base_type is None) or (base_type is type(None)):
        if strict and (data is not None):
            raise TypeError()
        return None
    elif base_type in {int, float, bool, str}:
        if strict and (not isinstance(data, (int, float, bool, str))):
            raise TypeError()
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
            if strict and (type(data) is not bytes):
                raise TypeError()
            return data
    elif base_type is list:
        if strict and (not isinstance(data, (list, tuple))):
            raise TypeError()

        # --- dataの方が多い場合、長さをそろえて上書きする
        # 短い場合はあるサイズまで上書きする
        if src_data is None:
            src_data = []
        assert isinstance(src_data, list)
        for i in range(len(data)):
            if i >= len(src_data):
                src_data.append(None)
        tp = get_args(type_)
        tp = Any if len(tp) == 0 else tp[0]
        for i in range(len(src_data)):
            if i >= len(data):
                break
            src_data[i] = _apply_dict_to_dataclass_sub(cast(type, tp), key, src_data[i], data[i], strict, depth + 1)
        return src_data
    elif (base_type is tuple) or issubclass(base_type, Sequence):
        if strict and (not isinstance(data, (list, tuple))):
            raise TypeError()

        # --- 長さは見ずにそのまま上書きする
        tps = get_args(type_)
        src_data = []
        for i in range(len(data)):
            if len(tps) == 1:
                tp = tps[0]
            elif i < len(tps):
                tp = tps[i]
            else:
                tp = Any
            src_data.append(_apply_dict_to_dataclass_sub(cast(type, tp), key, None, data[i], strict, depth + 1))
        return tuple(src_data)
    elif issubclass(base_type, enum.Enum):
        # --- strも変換
        if issubclass(type(data), enum.Enum):
            return data
        if data in base_type.__members__:
            return base_type[cast(Any, data)]
        if strict:
            raise TypeError()
        else:
            return data
    elif base_type is dict:
        if strict and (not isinstance(data, dict)):
            raise TypeError()

        # --- data側にあるkeyのみを上書きする
        if src_data is None:
            src_data = {}
        assert isinstance(src_data, dict)
        tp = get_args(type_)
        tp = Any if len(tp) == 0 else tp[1]
        for k in data.keys():
            src = src_data[k] if k in src_data else None
            src_data[k] = _apply_dict_to_dataclass_sub(cast(Any, tp), k, src, data[k], strict, depth + 1)
        return src_data
    elif is_dataclass(base_type):
        # --- dataclass
        if src_data is None:
            kwargs = {}
            for f in fields(base_type):
                if hasattr(data, f.name):
                    d = getattr(data, f.name)
                elif isinstance(data, dict) and (f.name in data):
                    d = data[f.name]
                else:
                    continue
                kwargs[f.name] = _apply_dict_to_dataclass_sub(cast(type, f.type), f.name, None, d, strict, depth + 1)
            return type_(**kwargs)
        else:
            for f in fields(base_type):
                if hasattr(data, f.name):
                    d = getattr(data, f.name)
                elif isinstance(data, dict) and (f.name in data):
                    d = data[f.name]
                else:
                    continue
                src = getattr(src_data, f.name) if hasattr(src_data, f.name) else None
                d = _apply_dict_to_dataclass_sub(cast(type, f.type), f.name, src, d, strict, depth + 1)
                setattr(src_data, f.name, d)
            return src_data
    elif base_type is np.ndarray:
        # --- to_list <-> np.ndarray
        if isinstance(data, np.ndarray):
            return data.copy()
        else:
            return np.array(data)
    elif isinstance(base_type, FunctionType):
        # 厳密なcopyは保留
        return data
    elif isinstance(base_type, object):
        if isinstance(data, dict) and hasattr(base_type, "from_dict"):
            # --- to_dict/from_dict がある場合は実行する
            if src_data is None:
                src_data = base_type()  # type: ignore
            src_data.from_dict(data)
            return src_data
        else:
            return pickle.loads(pickle.dumps(data))
    else:
        return pickle.loads(pickle.dumps(data))


def dataclass_to_dict(data: Any, exclude_names: List[str] = [], to_print: bool = False) -> Any:
    """
    dataclassを辞書形式に変換するユーティリティ関数

    Parameters
    ----------
    data : Any
        対象となるdataclassインスタンス。
        再帰的に内部のdataclassも辞書に変換する。
    exclude_names : list[str], optional
        辞書に含めたくないフィールド名のリスト。デフォルトは空リスト。
    to_print : bool, optional
        Trueの場合、print用の文字列を返す

    Returns
    -------
    Any
        辞書形式に変換されたデータ。
        dataclassでない場合はそのまま返す。

    Note
    ----
    - 再帰的に変換されるため、ネストされたdataclassにも対応。
    - 辞書やリスト、tupleの中の要素も再帰的に変換対象となる。
    """
    if data is None:
        return None
    elif type(data) in {int, float, bool, str}:
        return data
    elif isinstance(data, bytes):
        if to_print:
            try:
                return data.decode()
            except Exception:
                logger.info(traceback.format_exc())
                logger.warning(f"Decoding failed. Convert to string. {data}")
                return str(data)
        else:
            import base64

            return base64.b64encode(data).decode("utf-8")
    elif isinstance(data, list):
        return [dataclass_to_dict(d, [], to_print) for d in data]
    elif isinstance(data, tuple):
        return tuple(dataclass_to_dict(d, [], to_print) for d in data)
    elif isinstance(data, dict):
        return {
            k2: dataclass_to_dict(v2, [], to_print)  #
            for k2, v2 in data.items()
            if k2 not in exclude_names
        }
    elif issubclass(type(data), enum.Enum):
        return data.name
    elif is_dataclass(data):
        # dataclass -> dict
        d = {}
        if not to_print:
            d["_target_"] = f"{data.__class__.__module__}.{data.__class__.__name__}"  # type: ignore
        d.update(
            {
                f.name: dataclass_to_dict(getattr(data, f.name), [], to_print)  #
                for f in fields(data)
                if f.name not in exclude_names
            }
        )
        return d
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, FunctionType):
        if to_print:
            # __qualname__は相対パスなので使わない
            return f"{data.__module__}.{data.__name__}"
        else:
            return {"_target_": f"{data.__module__}.{data.__name__}"}
    elif isinstance(data, object):
        # to_dict/from_dict がある場合は実行する
        if hasattr(data, "to_dict"):
            d = {}
            if not to_print:
                d["_target_"] = f"{data.__class__.__module__}.{data.__class__.__name__}"
            d.update(
                {
                    k2: dataclass_to_dict(v2, [], to_print)  #
                    for k2, v2 in data.to_dict().items()  # type: ignore , to_dict OK
                    if k2 not in exclude_names
                }
            )
            return d
        else:
            if to_print:
                return str(f"{data.__class__.__module__}.{data.__class__.__name__}")
            else:
                return {"_target_": f"{data.__class__.__module__}.{data.__class__.__name__}"}
    else:
        if to_print:
            return str(data)
        else:
            return data


def get_modified_fields(obj: Any, exclude_names: List[str] = []) -> dict:
    d, _ = _get_modified_fields_sub(obj, None, exclude_names, "")
    return d


def _get_modified_fields_sub(a: Any, b: Any, exclude_names: List[str], parent_type: str) -> Tuple[Any, bool]:
    if isinstance(a, list) and isinstance(a, list):
        if len(a) != len(b):
            d = []
            for a2 in a:
                d2, _ = _get_modified_fields_sub(a2, None, [], "list")
                d.append(d2)
            return d, True
        else:
            diff = False
            d = []
            for a2, b2 in zip(a, b):
                d2, diff2 = _get_modified_fields_sub(a2, b2, [], "list")
                if diff2:
                    diff = True
                    d.append(d2)
            return d, diff
    elif isinstance(a, dict) and isinstance(b, dict):
        d = {}
        diff = False
        for k, v in a.items():
            if k not in b:
                diff = True
                b2 = None
            else:
                b2 = b[k]
            a2, diff2 = _get_modified_fields_sub(v, b2, [], "dict")
            if diff2:
                diff = True
                d[k] = a2
        return d, diff
    elif is_dataclass(a):
        a = cast(Any, a)
        b = a.__class__()
        d = {}
        if parent_type == "list":
            d["_target_"] = f"{a.__class__.__module__}.{a.__class__.__name__}"
        diff = False
        for f in fields(a):  # type: ignore
            if f.name in exclude_names:
                continue
            a2, diff2 = _get_modified_fields_sub(getattr(a, f.name), getattr(b, f.name), [], "dataclass")
            if diff2:
                diff = True
                d[f.name] = a2
        return d, diff

    elif a != b:
        return a, True
    else:
        return a, False  # diff
