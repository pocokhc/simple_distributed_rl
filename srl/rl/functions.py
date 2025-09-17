import random
from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from srl.base.define import SpaceTypes


def rescaling(x, eps=0.001):
    return np.sign(x) * (np.sqrt(np.abs(x) + 1.0) - 1.0) + eps * x


def inverse_rescaling(x, eps=0.001):
    n = np.sqrt(1.0 + 4.0 * eps * (np.abs(x) + 1.0 + eps)) - 1.0
    n = n / (2.0 * eps)
    return np.sign(x) * ((n**2) - 1.0)


def symlog(x):
    return np.sign(x) * np.log(1 + np.abs(x))


def symexp(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)


def signed_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def inverse_signed_sqrt(x):
    return np.sign(x) * (x**2)


def sqrt_symlog(x):
    abs_x = np.abs(x)
    sqrt = np.sign(x) * np.sqrt(abs_x)
    symlog = np.sign(x) * (np.log1p(abs_x - 1) + 1)
    return np.where(abs_x <= 1, sqrt, symlog)


def inverse_sqrt_symlog(x):
    abs_x = np.abs(x)
    square = np.sign(x) * (x**2)
    symexp = np.sign(x) * (np.exp(abs_x - 1))
    return np.where(abs_x <= 1, square, symexp)


def unimix(probs, unimix: float):
    uniform = np.ones_like(probs) / probs.shape[-1]
    return (1 - unimix) * probs + unimix * uniform


def sigmoid(x, a=1):
    return 1 / (1 + np.exp(-a * x))


def one_hot(x, size: int, dtype=np.float32):
    x = np.asarray(x)
    return np.identity(size, dtype=dtype)[x]


def twohot_encode(x, size: int, low: float, high: float, dtype=np.float32) -> np.ndarray:  # List[float]
    x = np.clip(x, a_min=low, a_max=high)
    # 0-bins のサイズで正規化
    x = (size - 1) * (x - low) / (high - low)
    # 整数部:idx 小数部:weight
    idx = np.floor(x).astype(np.int32)
    w = (x - idx)[..., np.newaxis]

    onehot = np.identity(size)
    onehot = np.vstack([onehot, np.zeros(size)])
    onehot1 = onehot[idx]
    onehot2 = onehot[idx + 1]
    return (onehot1 * (1 - w) + onehot2 * w).astype(dtype)


def twohot_decode(x: np.ndarray, size: int, low: float, high: float) -> np.ndarray:
    bins = np.arange(0, size)
    y = np.dot(x, bins)  # (batch,)
    y = (y / (size - 1)) * (high - low) + low
    return y


def create_beta_list(policy_num: int, max_beta=0.3):
    assert policy_num > 0
    beta_list = []
    for i in range(policy_num):
        if i == 0:
            b = 0
        elif i == policy_num - 1:
            b = max_beta
        else:
            b = 10 * (2 * i - (policy_num - 2)) / (policy_num - 2)
            b = max_beta * sigmoid(b)
        beta_list.append(b)
    return beta_list


def create_discount_list(policy_num: int, gamma0=0.9999, gamma1=0.997, gamma2=0.99):
    assert policy_num > 0
    discount_list = []
    for i in range(policy_num):
        if i == 0:
            g = gamma0
        elif 1 <= i and i <= 6:
            g = gamma0 + (gamma1 - gamma0) * sigmoid(10 * ((2 * i - 6) / 6))
        elif i == 7:
            g = gamma1
        else:
            g = (policy_num - 9 - (i - 8)) * np.log(1 - gamma1) + (i - 8) * np.log(1 - gamma2)
            g = 1 - np.exp(g / (policy_num - 9))
        discount_list.append(g)
    return discount_list


def create_epsilon_list(policy_num: int, epsilon=0.4, alpha=8.0):
    assert policy_num > 0
    if policy_num == 1:
        return [epsilon / 4]

    epsilon_list = []
    for i in range(policy_num):
        e = epsilon ** (1 + (i / (policy_num - 1)) * alpha)
        epsilon_list.append(e)
    return epsilon_list


def get_random_max_index(arr: Union[np.ndarray, List[float]], invalid_actions: List[int] = []) -> int:
    """Destructive to the original variable."""
    if len(arr) < 100:
        if len(invalid_actions) > 0:
            if isinstance(arr, np.ndarray):
                arr = arr.tolist()
            arr = arr[:]
            for a in invalid_actions:
                arr[a] = -np.inf
        max_value = max(arr)
        max_list = [i for i, val in enumerate(arr) if val == max_value]
        return max_list[0] if len(max_list) == 1 else random.choice(max_list)
    else:
        arr = np.asarray(arr, dtype=float)
        arr[invalid_actions] = -np.inf
        return random.choice(np.where(arr == arr.max())[0].tolist())


def get_random_idx_by_rankbase(arr_size: int, alpha: float = 1.0):
    total = arr_size * (2 + (arr_size - 1) * alpha) / 2
    r = random.random() * total
    inverse_r = (alpha - 2 + np.sqrt((2 - alpha) ** 2 + 8 * alpha * r)) / (2 * alpha)
    idx = int(inverse_r)
    return idx


def random_choice_by_probs(probs, total=None):
    if total is None:
        total = sum(probs)
    r = random.random() * total

    num = 0
    for i, weight in enumerate(probs):
        num += weight
        if r <= num:
            return i

    raise ValueError(f"not coming. total: {total}, r: {r}, num: {num}, probs: {probs}")


def calc_epsilon_greedy_probs(q, invalid_actions, epsilon, action_num):
    # filter
    q = np.array([(-np.inf if a in invalid_actions else v) for a, v in enumerate(q)])

    q_max = np.amax(q, axis=0)
    q_max_num = np.count_nonzero(q == q_max)

    valid_action_num = action_num - len(invalid_actions)
    probs = []
    for a in range(action_num):
        if a in invalid_actions:
            probs.append(0.0)
        else:
            prob = epsilon / valid_action_num
            if q[a] == q_max:
                prob += (1 - epsilon) / q_max_num
            probs.append(prob)
    return probs


def create_fancy_index_for_invalid_actions(idx_list: List[List[int]]):
    """ファンシーインデックス
    idx_list = [
        [1, 2, 5],
        [2],
        [2, 3],
    ]
    idx1 = [0, 0, 0, 1, 2, 2]
    idx2 = [1, 2, 5, 2, 2, 3]
    """
    idx1 = [i for i, sublist in enumerate(idx_list) for _ in sublist]
    idx2 = [item for sublist in idx_list for item in sublist]
    return idx1, idx2


def image_processor(
    rgb_array: np.ndarray,  # (H,W,C)
    from_space_type: SpaceTypes,
    to_space_type: SpaceTypes,
    resize: Optional[Tuple[int, int]] = None,  # resize: (w, h)
    trimming: Optional[Tuple[int, int, int, int]] = None,  # (top, left, bottom, right)
    normalize_type: Literal["", "0to1", "-1to1"] = "",
    shape_order: Literal["HWC", "CHW"] = "HWC",  # "HWC": tf(H,W,C), "CHW": torch(C,H,W)
):
    assert from_space_type in [
        SpaceTypes.GRAY_2ch,
        SpaceTypes.GRAY_3ch,
        SpaceTypes.COLOR,
    ]
    import cv2

    if to_space_type == SpaceTypes.COLOR and (from_space_type == SpaceTypes.GRAY_2ch or from_space_type == SpaceTypes.GRAY_3ch):
        # gray -> color
        rgb_array = cv2.applyColorMap(rgb_array, cv2.COLORMAP_HOT)
    elif from_space_type == SpaceTypes.COLOR and (to_space_type == SpaceTypes.GRAY_2ch or to_space_type == SpaceTypes.GRAY_3ch):
        # color -> gray
        rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

    if trimming is not None:
        top, left, bottom, right = trimming
        assert top < bottom
        assert left < right
        w = rgb_array.shape[1]
        h = rgb_array.shape[0]
        if top < 0:
            top = 0
        if left < 0:
            left = 0
        if bottom > h:
            bottom = h
        if right > w:
            right = w
        rgb_array = rgb_array[top:bottom, left:right]

    w = rgb_array.shape[1]
    h = rgb_array.shape[0]
    if resize is not None and (w, h) != resize:
        rgb_array = cv2.resize(rgb_array, resize)

    if from_space_type == SpaceTypes.GRAY_3ch and to_space_type == SpaceTypes.GRAY_2ch and len(rgb_array.shape) == 3:
        rgb_array = np.squeeze(rgb_array, axis=-1)
    elif len(rgb_array.shape) == 2 and to_space_type == SpaceTypes.GRAY_3ch:
        rgb_array = rgb_array[..., np.newaxis]

    if normalize_type == "0to1":
        rgb_array = rgb_array.astype(np.float32)
        rgb_array /= 255.0
    elif normalize_type == "-1to1":
        rgb_array = rgb_array.astype(np.float32)
        rgb_array = (rgb_array * 2.0 / 255.0) - 1.0

    if len(rgb_array.shape) == 3 and shape_order == "CHW":
        rgb_array = np.transpose(rgb_array, (2, 0, 1))

    return rgb_array
