import pytest

import srl.rl.torch_.converter as conv


@pytest.mark.parametrize(
    "activation",
    [
        "sigmoid",
        "softmax",
        "tanh",
        "relu",
        "leaky_relu",
        "elu",
        "swish",
        "mish",
    ],
)
def test_convert_activation(activation):
    pytest.importorskip("torch")

    act = conv.convert_activation_torch(activation)
    print(act)


@pytest.mark.parametrize(
    "initializer",
    [
        "zeros",
        "ones",
        "he_normal",
        "glorot_uniform",
        # "truncated_normal",
    ],
)
def test_convert_initializer(initializer):
    pytest.importorskip("torch")
    import torch

    x = torch.randn([2, 4], dtype=torch.float32)
    initializer = conv.apply_initializer_torch(x, initializer)
    print(initializer)
