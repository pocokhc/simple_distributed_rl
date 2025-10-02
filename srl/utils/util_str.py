from typing import Union

import numpy as np


def to_str_time(sec: float) -> str:
    if sec == np.inf:
        return "   inf"
    if sec < 60:
        return "{:5.2f}s".format(sec)
    if sec < 60 * 60 * 2:
        return "{:5.1f}m".format(sec / 60)
    return "{:5.2f}h".format(sec / (60 * 60))


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
