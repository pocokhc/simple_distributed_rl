import numpy as np


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
