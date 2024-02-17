import random
from typing import List, Union

import numpy as np

from srl.base.define import EnvObservationTypes


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


def sigmoid(x, a=1):
    return 1 / (1 + np.exp(-a * x))


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
        return random.choice(np.where(arr == arr.max())[0].tolist())


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


def to_str_observation(state: np.ndarray, state_type: EnvObservationTypes = EnvObservationTypes.UNKNOWN) -> str:
    if state_type == EnvObservationTypes.UNKNOWN:
        pass
    else:
        state = state.astype(
            {
                EnvObservationTypes.DISCRETE: np.int32,
                EnvObservationTypes.CONTINUOUS: np.float32,
                EnvObservationTypes.GRAY_2ch: np.int32,
                EnvObservationTypes.GRAY_3ch: np.int32,
                EnvObservationTypes.COLOR: np.int32,
                EnvObservationTypes.IMAGE: np.int32,
            }[state_type]
        )
    return str(state.flatten().tolist()).replace(" ", "")[1:-1]


def render_discrete_action(maxa, env, config, func) -> None:
    action_num = config.action_num
    invalid_actions = env.get_invalid_actions()
    for action in range(action_num):
        if action in invalid_actions:
            continue
        s = ""
        if maxa is not None:
            if action == maxa:
                s += "*"
            else:
                s += " "
        rl_s = func(action)
        s += f"{env.action_to_str(action):3s}: {rl_s}"
        print(s)

    # invalid actions
    view_invalid_actions_num = 0
    for action in range(action_num):
        if action not in invalid_actions:
            continue
        if view_invalid_actions_num > 2:
            continue
        s = "x"
        view_invalid_actions_num += 1
        rl_s = func(action)
        s += f"{env.action_to_str(action):3s}: {rl_s}"
        print(s)
    if view_invalid_actions_num > 2:
        print("... Some invalid actions have been omitted.")


def twohot_encode(x, size: int, low: float, high: float) -> np.ndarray:  # List[float]
    x = np.clip(x, a_min=low, a_max=high)
    # 0-bins のサイズで正規化
    x = (size - 1) * (x - low) / (high - low)
    # 整数部:idx 小数部:weight
    idx = np.floor(x).astype(np.int32)
    w = (x - idx)[..., np.newaxis]

    onehot = np.identity(size, dtype=np.float32)
    onehot = np.vstack([onehot, np.zeros(size)])
    onehot1 = onehot[idx]
    onehot2 = onehot[idx + 1]
    return onehot1 * (1 - w) + onehot2 * w


def twohot_decode(x, size: int, low: float, high: float):
    bins = np.arange(0, size)
    x = np.dot(x, bins)
    return (x / (size - 1)) * (high - low) + low
