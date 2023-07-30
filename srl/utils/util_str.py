from typing import List, Optional, Union

import numpy as np

from srl.base.define import InfoType


def to_str_time(sec: float) -> str:
    if sec == np.inf:
        return "   inf"
    if sec < 60:
        return "{:5.2f}s".format(sec)
    return "{:5.1f}m".format(sec / 60)


def to_str_reward(reward: Union[int, float], check_skip: bool = False) -> str:
    if check_skip:
        is_int = False
    else:
        if isinstance(reward, int):
            is_int = True
        elif reward.is_integer():
            is_int = True
        else:
            is_int = False

    if is_int:
        return "{:2d}".format(int(reward))
    else:
        if -10 <= reward <= 10:
            return "{:6.3f}".format(float(reward))
        else:
            return "{:6.1f}".format(float(reward))


def to_str_info(info: Optional[InfoType], types={}) -> str:
    if info is None:
        return ""

    # check type
    key_types = {}
    for k, v in info.items():
        t = types.get(k, {}).get("type", None)
        if t is None:
            # check
            if isinstance(v, int):
                t = int
            elif isinstance(v, float):
                t = float
            elif isinstance(v, np.integer):
                t = int
            elif isinstance(v, np.floating):
                t = float
            elif isinstance(v, np.ndarray):
                t = float
        key_types[k] = t

    s = ""
    for k, v in info.items():
        t = key_types[k]
        if t == int:
            s += f"|{k} {int(v):2d}"
        elif t == float:
            v = float(v)
            if -10 <= v <= 10:
                s += f"|{k} {v:.3f}"
            else:
                s += f"|{k} {v:.1f}"
        else:
            s += f"|{k} {str(v)}"

    return s


def to_str_from_list_info(infos: List[InfoType], types={}) -> str:
    info2 = {}
    for info in infos:
        if info is None:
            continue
        for k, v in info.items():
            if k not in info2:
                info2[k] = [v]
            else:
                info2[k].append(v)

    s = ""
    for k, arr in info2.items():
        t = types.get(k, {}).get("type", None)
        d = types.get(k, {}).get("data", "ave")
        if t == int:
            arr = [int(a) for a in arr]
        elif t == float:
            arr = [float(a) for a in arr]
        elif t == str:
            # strは最後の値を採用
            s += f"|{k} {str(arr[-1])}"
            continue
        else:
            # check
            t = int
            vals = []
            for v in arr:
                if isinstance(v, int):
                    vals.append(v)
                elif isinstance(v, float):
                    vals.append(v)
                    t = float
                elif isinstance(v, np.integer):
                    vals.append(int(v))
                elif isinstance(v, np.floating):
                    vals.append(float(v))
                    t = float
                elif isinstance(v, np.ndarray):
                    vals.append(float(v))
                    t = float
            if len(vals) == 0:
                s += f"|{k} 0"
                continue
            else:
                arr = vals
        # ---
        if d == "last":
            v = arr[-1]
        elif d == "min":
            v = min(arr)
        elif d == "max":
            v = max(arr)
        else:
            v = np.mean(arr)
            t = float
        # ---
        if t == int:
            s += f"|{k} {v:2d}"
        else:
            if -10 <= v <= 10:
                s += f"|{k} {v:.3f}"
            else:
                s += f"|{k} {v:.1f}"
    return s
