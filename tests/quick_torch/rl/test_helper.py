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


def test_twohot():
    pytest.importorskip("torch")
    import torch

    from srl.rl.torch_ import helper

    x = torch.FloatTensor([2.4, -2.6])
    cat = helper.twohot_encode(x, 11, -5, 5, "cpu")
    _cat = cat.cpu().numpy()
    print(_cat)
    assert pytest.approx(_cat[0][7]) == 0.6
    assert pytest.approx(_cat[0][8]) == 0.4
    assert pytest.approx(_cat[1][2]) == 0.6
    assert pytest.approx(_cat[1][3]) == 0.4

    val = helper.twohot_decode(cat, 11, -5, 5, "cpu")
    _val = val.cpu().numpy()
    print(_val)
    assert pytest.approx(_val[0]) == 2.4
    assert pytest.approx(_val[1]) == -2.6

    # out range
    x = torch.FloatTensor([7, -7])
    cat = helper.twohot_encode(x, 5, -2, 2, "cpu")
    _cat = cat.cpu().numpy()
    print(_cat)
    assert pytest.approx(_cat[0][3]) == 0.0
    assert pytest.approx(_cat[0][4]) == 1.0
    assert pytest.approx(_cat[1][0]) == 1.0
    assert pytest.approx(_cat[1][1]) == 0.0

    val = helper.twohot_decode(cat, 5, -2, 2, "cpu")
    _val = val.cpu().numpy()
    print(_val)
    assert pytest.approx(val[0]) == 2
    assert pytest.approx(val[1]) == -2
