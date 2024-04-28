import pytest


def test_twohot():
    pytest.importorskip("torch")
    import torch

    from srl.rl.torch_ import functions

    x = torch.FloatTensor([2.4, -2.6])
    cat = functions.twohot_encode(x, 11, -5, 5, "cpu")
    _cat = cat.cpu().numpy()
    print(_cat)
    assert pytest.approx(_cat[0][7]) == 0.6
    assert pytest.approx(_cat[0][8]) == 0.4
    assert pytest.approx(_cat[1][2]) == 0.6
    assert pytest.approx(_cat[1][3]) == 0.4

    val = functions.twohot_decode(cat, 11, -5, 5, "cpu")
    _val = val.cpu().numpy()
    print(_val)
    assert pytest.approx(_val[0]) == 2.4
    assert pytest.approx(_val[1]) == -2.6

    # out range
    x = torch.FloatTensor([7, -7])
    cat = functions.twohot_encode(x, 5, -2, 2, "cpu")
    _cat = cat.cpu().numpy()
    print(_cat)
    assert pytest.approx(_cat[0][3]) == 0.0
    assert pytest.approx(_cat[0][4]) == 1.0
    assert pytest.approx(_cat[1][0]) == 1.0
    assert pytest.approx(_cat[1][1]) == 0.0

    val = functions.twohot_decode(cat, 5, -2, 2, "cpu")
    _val = val.cpu().numpy()
    print(_val)
    assert pytest.approx(val[0]) == 2
    assert pytest.approx(val[1]) == -2
