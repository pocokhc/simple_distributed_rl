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


def apply_initializer_torch(x, initializer: str):
    """
    指定された initializer を使用してテンソル x を初期化する。

    :param x: 初期化対象のテンソル
    :param initializer: 初期化関数名（例: "he_normal", "glorot_uniform"）
    :return: 初期化後のテンソル
    """

    import torch
    import torch.nn.init as init

    with torch.no_grad():
        initializer = initializer.lower()

        if "he_normal" == initializer:
            return init.kaiming_normal_(x, mode="fan_in", nonlinearity="relu")

        if "glorot_uniform" == initializer:
            return init.xavier_uniform_(x)

        # init モジュール内の関数を検索
        available_inits = {name.lower().replace("_", ""): name for name in dir(init) if not name.startswith("_")}

        if initializer in available_inits:
            return getattr(init, available_inits[initializer])(x)

    raise ValueError(f"Unknown initializer: {initializer}")
