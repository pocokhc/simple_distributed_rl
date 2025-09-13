import math
import time

import numpy as np
from tqdm import tqdm


def speed_test_torch():
    import torch

    from srl.rl.torch_.functions import inverse_rescaling, inverse_signed_sqrt, inverse_sqrt_symlog, rescaling, signed_sqrt, sqrt_symlog, symexp, symlog

    epochs = 100_000
    x = torch.tensor(np.full((128, 4), 2))

    for name, func, inv_func in [
        ["rescaling", rescaling, inverse_rescaling],
        ["symlog", symlog, symexp],
        ["sqrt", signed_sqrt, inverse_signed_sqrt],
        ["sqrt_symlog", sqrt_symlog, inverse_sqrt_symlog],
    ]:
        t0 = time.time()
        for _ in tqdm(range(epochs)):
            y = func(x)
        print(name, "forward", time.time() - t0)

        t0 = time.time()
        for _ in tqdm(range(epochs)):
            _ = inv_func(y)
        print(name, "inverse", time.time() - t0)


def plot():
    import matplotlib.pyplot as plt

    from srl.rl.functions import rescaling, sqrt_symlog, symlog

    def symlog_scalar(x: float, shift: float = 1) -> float:
        if -shift <= x <= shift:
            return x
        return math.copysign(math.log1p(abs(x) - shift) + shift, x)

    x_vals = np.linspace(-3, 3, 10000)

    y_scalar = np.array([symlog_scalar(x) for x in x_vals])
    y_symlog = np.array([symlog(x) for x in x_vals])
    y_rescaling = np.array([rescaling(x) for x in x_vals])
    y_sqrt_symlog = np.array([sqrt_symlog(x) for x in x_vals])

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, x_vals, label="x")
    plt.plot(x_vals, y_scalar, label="symlog_scalar")
    plt.plot(x_vals, y_symlog, label="symlog")
    plt.plot(x_vals, y_rescaling, label="rescaling")
    plt.plot(x_vals, y_sqrt_symlog, label="sqrt_symlog")
    plt.axvline(-1, color="gray", linestyle=":")
    plt.axvline(1, color="gray", linestyle=":")
    plt.title("Scalar Symlog Function")
    plt.xlabel("x")
    plt.ylabel("output")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    speed_test_torch()
    plot()
