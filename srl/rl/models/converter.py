def convert_activation_tf(activation: str):
    return activation


def convert_activation_torch(activation: str):
    import torch.nn as nn

    if activation.lower() == "swish":
        return nn.SiLU

    if hasattr(nn, activation):
        return getattr(nn, activation)

    for name in dir(nn):
        if name.startswith("_"):
            continue
        _a = activation.lower()
        _a = _a.replace("_", "")
        if _a == name.lower():
            return getattr(nn, name)

    raise ValueError(activation)
