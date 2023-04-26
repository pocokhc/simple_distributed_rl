import pytest

from srl.rl.functions.common import float_category_decode, float_category_encode


def test_category():
    # plus
    cat = float_category_encode(2.4, -5, 5)
    assert pytest.approx(cat[7]) == 0.6
    assert pytest.approx(cat[8]) == 0.4

    val = float_category_decode(cat, -5, 5)
    assert pytest.approx(val) == 2.4

    # minus
    cat = float_category_encode(-2.6, -5, 5)
    assert pytest.approx(cat[2]) == 0.6
    assert pytest.approx(cat[3]) == 0.4

    val = float_category_decode(cat, -5, 5)
    assert pytest.approx(val) == -2.6

    # out range(plus)
    cat = float_category_encode(7, -2, 2)
    assert pytest.approx(cat[3]) == 0.0
    assert pytest.approx(cat[4]) == 1.0

    val = float_category_decode(cat, -2, 2)
    assert pytest.approx(val) == 2

    # "out range(minus)"
    cat = float_category_encode(-7, -2, 2)
    assert pytest.approx(cat[0]) == 1.0
    assert pytest.approx(cat[1]) == 0.0

    val = float_category_decode(cat, -2, 2)
    assert pytest.approx(val) == -2
