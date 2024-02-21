import numpy as np
import pytest

from srl.base.define import RLTypes
from srl.base.spaces import ContinuousSpace


def test_space():
    space = ContinuousSpace(-1, 3)
    assert space.rl_type == RLTypes.CONTINUOUS

    print(space)

    # --- continuous list
    assert space.list_size == 1
    np.testing.assert_array_equal(space.list_low, [-1])
    np.testing.assert_array_equal(space.list_high, [3])
    de = space.decode_from_list_float([1.2])
    assert isinstance(de, float)
    assert de == 1.2
    en = space.encode_to_list_float(1.2)
    assert isinstance(en, list)
    for n in en:
        assert isinstance(n, float)
    assert en[0] == 1.2

    # --- continuous numpy
    assert space.shape == (1,)
    np.testing.assert_array_equal(space.low, [-1])
    np.testing.assert_array_equal(space.high, [3])
    de = space.decode_from_np(np.array([1.2]))
    assert isinstance(de, float)
    assert de == 1.2
    en = space.encode_to_np(1.2, np.float32)
    np.testing.assert_array_equal(en, np.array([1.2], dtype=np.float32))

    # --- sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, float)
        assert action >= -1
        assert action <= 3

    # --- eq
    assert space == ContinuousSpace(-1, 3)
    assert space != ContinuousSpace(-1, 2)


def test_discrete_no_division():
    space = ContinuousSpace(-1, 3)

    # --- discrete
    with pytest.raises(AssertionError):
        _ = space.n
    with pytest.raises(AssertionError):
        _ = space.encode_to_int(1.2)
    de = space.decode_from_int(2)
    assert de == 2.0

    # --- discrete
    en = space.encode_to_list_int(1.2)
    assert len(en) == 1
    assert en[0] == 1
    de = space.decode_from_list_int([1])
    assert de == 1.0


def test_discrete_division():
    space = ContinuousSpace(-1, 3)

    # action discrete
    space.create_division_tbl(5)
    assert space.division_tbl is not None
    true_tbl = [
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
    ]
    np.testing.assert_array_equal(true_tbl, space.division_tbl)

    # --- discrete
    assert space.n == 5
    en = space.encode_to_int(1.2)
    assert en == 2
    de = space.decode_from_int(2)
    assert de == 1.0

    # --- discrete numpy
    en = space.encode_to_list_int(1.2)
    assert len(en) == 1
    assert en[0] == 2
    de = space.decode_from_list_int([2])
    assert de == 1.0


def test_inf():
    space = ContinuousSpace()

    print(space)

    # sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, float)

    # --- discrete
    space.create_division_tbl(5)
    with pytest.raises(AssertionError):
        _ = space.encode_to_int(1.2)

    # --- continuous list
    assert space.list_size == 1
    np.testing.assert_array_equal(space.list_low, [-np.inf])
    np.testing.assert_array_equal(space.list_high, [np.inf])


def test_sanitize():
    space = ContinuousSpace(-1, 3)

    val = space.sanitize(1)
    assert space.check_val(val)
    assert val == 1.0

    val = space.sanitize([2])
    assert space.check_val(val)
    assert val == 2.0

    val = space.sanitize((1,))
    assert space.check_val(val)
    assert val == 1.0

    assert not space.check_val(1)
    assert not space.check_val(3.1)
    assert not space.check_val(-1.1)


def test_sanitize2():
    space = ContinuousSpace(-1, 3)

    val = space.sanitize([-2])
    assert space.check_val(val)
    assert val == -1

    val = space.sanitize((6, 99))
    assert space.check_val(val)
    assert val == 3
