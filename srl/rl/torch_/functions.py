import torch

"""
torchに依存していない処理
torchに関する処理を助けるライブラリ群はhelperへ
"""


def rescaling(x, eps=0.001):
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + eps * x


def inverse_rescaling(x, eps=0.001):
    n = torch.sqrt(1.0 + 4.0 * eps * (torch.abs(x) + 1.0 + eps)) - 1.0
    n = n / (2.0 * eps)
    return torch.sign(x) * ((n**2) - 1.0)


def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def unimix(probs, unimix: float):
    uniform = torch.ones_like(probs) / probs.shape[-1]
    return (1 - unimix) * probs + unimix * uniform


def twohot_encode(x: torch.Tensor, size: int, low: float, high: float, device) -> torch.Tensor:
    x = x.clamp(low, high)
    # 0-bins のサイズで正規化
    x = (size - 1) * (x - low) / (high - low)
    # 整数部:idx 小数部:weight
    idx = x.floor().to(torch.int32)
    w = (x - idx).unsqueeze(-1)

    onehot = torch.eye(size, dtype=torch.float32).to(device)
    onehot = torch.vstack([onehot, torch.zeros(size)])
    onehot1 = onehot[idx]
    onehot2 = onehot[idx + 1]
    return onehot1 * (1 - w) + onehot2 * w


def twohot_decode(x: torch.Tensor, size: int, low: float, high: float, device):
    bins = torch.arange(0, size).to(device)
    bins = bins.unsqueeze(0).tile((x.shape[0], 1))
    x = x * bins
    x = x.sum(1)
    return (x / (size - 1)) * (high - low) + low


def binary_onehot_decode(x: torch.Tensor):
    return x[:, 0]
