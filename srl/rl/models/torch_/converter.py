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


def set_initializer_torch(x, initializer: str):
    import torch.nn.init as init

    initializer = initializer.lower()

    if "he_normal" == initializer:
        return init.kaiming_normal_(x, mode="fan_in", nonlinearity="relu")

    if "glorot_uniform" == initializer:
        return init.xavier_uniform_(x)

    if hasattr(init, initializer):
        return getattr(init, initializer)(x)

    for name in dir(init):
        if name.startswith("_"):
            continue
        name2 = name.replace("_", "")
        if initializer == name2.lower():
            return getattr(init, name)(x)

    raise ValueError(initializer)
