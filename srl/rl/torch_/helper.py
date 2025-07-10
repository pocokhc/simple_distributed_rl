import copy
from typing import Iterable

import torch
import torch.nn as nn

"""
torchに関する処理を助けるライブラリ群
torchに依存していない処理はfunctionsへ
"""


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


def model_params_soft_sync(target_params: Iterable[nn.Parameter], source_params: Iterable[nn.Parameter], tau: float):
    """モデルのパラメータをsoft update（指数移動平均）で同期する"""
    for src_param, tgt_param in zip(source_params, target_params):
        tgt_param.data.copy_(tau * src_param.data + (1.0 - tau) * tgt_param.data)


def reset_model_params(params: Iterable[nn.Parameter], rate: float, sigma: float = 2.0):
    if sigma > 0:
        for param in params:
            param.data.copy_(rate * param.data + torch.randn_like(param) * sigma)
    else:
        for param in params:
            param.data.copy_(rate * param.data)


def model_restore(model: nn.Module, dat, from_serialized: bool = False):
    if from_serialized:
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


def model_backup(model: nn.Module, serialized: bool = False):
    # backupでdeviceを変えるとtrainerとの並列処理でバグの可能性あり
    if serialized:
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
