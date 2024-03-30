import torch


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
