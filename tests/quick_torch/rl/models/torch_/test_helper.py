import numpy as np
import pytest


def test_model_soft_sync():
    pytest.importorskip("torch")
    import torch

    from srl.rl.torch_ import helper

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(1, 4, bias=False)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return x

    m1 = Model()
    m2 = Model()
    torch.nn.init.zeros_(m1.fc1.weight)
    torch.nn.init.ones_(m2.fc1.weight)

    for p in m1.parameters():
        print(p)
    for p in m2.parameters():
        print(p)

    helper.model_soft_sync(m1, m2, 0.1)
    for p in m1.parameters():
        print(p)
    for p in m2.parameters():
        print(p)
    assert (m1.fc1.weight.detach().numpy() == np.array([0.1, 0.1, 0.1, 0.1], np.float32)).all()
    assert (m2.fc1.weight.detach().numpy() == np.array([1, 1, 1, 1])).all()
