import numpy as np
import pytest


def test_rescaling():
    pytest.importorskip("torch")
    import torch

    from srl.rl.torch_ import functions

    x = torch.tensor(np.linspace(-100, 100, 10000, dtype=np.float32))
    y = functions.rescaling(x)
    x2 = functions.inverse_rescaling(y)
    print(x)
    print(x2)
    assert torch.allclose(x, x2, rtol=1e-3, atol=1e-3)


def test_symlog():
    pytest.importorskip("torch")
    import torch

    from srl.rl.torch_ import functions

    x = torch.tensor(np.linspace(-100, 100, 10000, dtype=np.float32))
    y = functions.symlog(x)
    x2 = functions.symexp(y)
    print(x)
    print(x2)
    assert torch.allclose(x, x2, rtol=1e-3, atol=1e-3)


def test_signed_sqrt():
    pytest.importorskip("torch")
    import torch

    from srl.rl.torch_ import functions

    x = torch.tensor(np.linspace(-100, 100, 10000, dtype=np.float32))
    y = functions.signed_sqrt(x)
    x2 = functions.inverse_signed_sqrt(y)
    print(x)
    print(x2)
    assert torch.allclose(x, x2, rtol=1e-3, atol=1e-3)


def test_sqrt_symlog():
    pytest.importorskip("torch")
    import torch

    from srl.rl.torch_ import functions

    x = torch.tensor(np.linspace(-100, 100, 10000, dtype=np.float32))
    y = functions.sqrt_symlog(x)
    x2 = functions.inverse_sqrt_symlog(y)
    print(x)
    print(x2)
    assert torch.allclose(x, x2, rtol=1e-3, atol=1e-3)
