import copy

import torch
import torch.nn as nn


def unimix(probs, unimix: float):
    uniform = torch.ones_like(probs) / probs.shape[-1]
    return (1 - unimix) * probs + unimix * uniform


def encode_sequence_batch(x: torch.Tensor):
    # (batch, seq, shape) -> (batch*seq, shape)
    size = x.size()
    head_size1 = size[0]
    head_size2 = size[1]
    shape = size[2:]
    x = x.reshape((head_size1 * head_size2, *shape))
    return x, head_size1, head_size2


def decode_sequence_batch(x: torch.Tensor, head_size1: int, head_size2: int):
    # (batch*seq, shape) -> (batch, seq, shape)
    size = x.size()
    shape = size[1:]
    x = x.view(head_size1, head_size2, *shape)
    return x


def model_sync(target_model: nn.Module, source_model: nn.Module):
    target_model.load_state_dict(source_model.state_dict())


def model_soft_sync(target_model: nn.Module, source_model: nn.Module, tau: float):
    for param, target_param in zip(source_model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def model_restore(model: nn.Module, dat, from_cpu: bool = False):
    if from_cpu:
        try:
            used_device = next(model.parameters()).device
        except StopIteration:
            # no params
            return
        if "cpu" in str(used_device):
            model.load_state_dict(dat)
        else:
            model.to("cpu").load_state_dict(dat)
            model.to(used_device)
    else:
        model.load_state_dict(dat)


def model_backup(model: nn.Module, to_cpu: bool = False):
    # backupでdeviceを変えるとtrainerとの並列処理でバグの可能性あり
    if to_cpu:
        try:
            used_device = next(model.parameters()).device
        except StopIteration:
            # no params
            return
        if "cpu" in str(used_device):
            return model.state_dict()
        else:
            return copy.deepcopy(model).to("cpu").state_dict()
    else:
        return model.state_dict()


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
