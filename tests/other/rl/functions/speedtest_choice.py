import random
import time

import numpy as np


def speedtest_choice():
    for size in [
        10,
        100,
        1_000,
        10_000,
    ]:
        for is_np in [False, True]:
            print(f"size={size}, numpy={is_np}")
            for name, func in [
                ["random", random.choice],
                ["numpy", np.random.choice],
            ]:
                _speedtest_choice(name, func, size, is_np)


def _speedtest_choice(name, func, size, is_np):
    arr = [i for i in range(size)]
    if is_np:
        arr = np.array(arr)

    t0 = time.time()
    for _ in range(100_000):
        action = func(arr)
    print("{:10s}: {:.5f}s".format(name, time.time() - t0))


if __name__ == "__main__":
    speedtest_choice()
