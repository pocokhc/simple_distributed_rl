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


def signed_sqrt(x):
    return torch.sign(x) * torch.sqrt(torch.abs(x))


def inverse_signed_sqrt(x):
    return torch.sign(x) * (x**2)


def sqrt_symlog(x: torch.Tensor):
    abs_x = x.abs()
    sqrt = x.sign() * torch.sqrt(abs_x)
    symlog = x.sign() * (torch.log1p(abs_x - 1) + 1)
    return torch.where(abs_x <= 1, sqrt, symlog)


def inverse_sqrt_symlog(x: torch.Tensor):
    abs_x = x.abs()
    square = x.sign() * (x**2)
    symexp = x.sign() * (torch.exp(abs_x - 1))
    return torch.where(abs_x <= 1, square, symexp)


def linear_symlog(x: torch.Tensor):
    abs_x = x.abs()
    symlog = x.sign() * (torch.log1p(abs_x - 1) + 1)
    return torch.where(abs_x <= 1, x, symlog)


def inverse_linear_symlog(x: torch.Tensor):
    abs_x = x.abs()
    symexp = x.sign() * (torch.exp(abs_x - 1))
    return torch.where(abs_x <= 1, x, symexp)


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
