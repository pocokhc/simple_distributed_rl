import random
import time

import numpy as np


def speedtest():
    for size in [
        10,
        50,
        100,
        500,
    ]:
        print(f"--------- size {size} ---------")
        for is_np in [False, True]:
            for is_choice in [False, True]:
                for is_invalid in [False, True]:
                    print(f"numpy={is_np}, choice={is_choice}, is_invalid={is_invalid}")
                    for name, func in [
                        ["numpy", _numpy],
                        ["numpy_random", _numpy_random],
                        ["random", _random],
                        ["random2", _random2],
                        ["random3", _random3],
                    ]:
                        _speedtest(name, func, size, is_np, is_choice, is_invalid)


def _numpy(arr, invalid_actions):
    arr = np.asarray(arr, dtype=float)
    arr[invalid_actions] = -np.inf
    return np.random.choice(np.where(arr == arr.max())[0])


def _numpy_random(arr, invalid_actions):
    arr = np.asarray(arr, dtype=float)
    arr[invalid_actions] = -np.inf
    return random.choice(np.where(arr == arr.max())[0].tolist())


def _random(arr, invalid_actions):
    if len(invalid_actions) > 0:
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        for a in invalid_actions:
            arr[a] = -np.inf
    max_value = max(arr)
    max_list = [i for i, val in enumerate(arr) if val == max_value]
    random_index = random.choice(max_list)
    return random_index


def _random2(arr, invalid_actions):
    if len(invalid_actions) > 0:
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        for a in invalid_actions:
            arr[a] = -np.inf
    max_value = max(arr)
    max_list = [i for i, val in enumerate(arr) if val == max_value]
    return max_list[0] if len(max_list) == 1 else random.choice(max_list)


def _random3(arr, invalid_actions):
    arr2 = []
    m = 0
    for i, n in enumerate(arr):
        if i in invalid_actions:
            continue
        if n == m:
            arr2.append(i)
        elif n > m:
            arr2 = [i]
            m = n
    return random.choice(arr2)


def _speedtest(name, func, size, is_np, is_choice, is_invalid):
    try_times = 100_000

    if is_choice:
        arr = [5.2 for i in range(size)]
    else:
        arr = [i for i in range(size)]
    if is_invalid:
        invalid_actions = [i for i in range(size) if i % 2 == 0]
    else:
        invalid_actions = []

    np_time = 0
    if is_np:
        t0 = time.time()
        for _ in range(try_times):
            arr = np.array(arr)
        np_time = time.time() - t0

    t0 = time.time()
    for _ in range(try_times):
        _ = func(arr, invalid_actions)
    print("{:20s}: {:.5f}s (+np time {:.5f}s)".format(name, time.time() - t0, np_time))


if __name__ == "__main__":
    speedtest()
