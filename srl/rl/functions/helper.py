import random
from typing import List, Union

import numpy as np

from srl.base.env.env_run import EnvRun


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


def render_discrete_action(maxa: int, action_num: int, env: EnvRun, func) -> None:
    invalid_actions = env.get_invalid_actions()
    for action in range(action_num):
        if action in invalid_actions:
            continue
        s = "*" if action == maxa else " "
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
