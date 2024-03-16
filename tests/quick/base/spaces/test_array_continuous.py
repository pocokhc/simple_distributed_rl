import numpy as np
import pytest

from srl.base.define import EnvTypes
from srl.base.spaces import ArrayContinuousSpace


def test_space():
    space = ArrayContinuousSpace(3, -1, 3)
    print(space)
    assert space.env_type == EnvTypes.CONTINUOUS

    # --- continuous list
    assert space.list_size == 3
    np.testing.assert_array_equal(space.list_low, [-1, -1, -1])
    np.testing.assert_array_equal(space.list_high, [3, 3, 3])
    de = space.decode_from_list_float([1.2, 1.2, 1.2])
    assert isinstance(de, list)
    for n in de:
        assert isinstance(n, float)
    np.testing.assert_array_equal(de, [1.2, 1.2, 1.2])
    en = space.encode_to_list_float([1.2, 1.2, 1.2])
    assert isinstance(en, list)
    for n in en:
        assert isinstance(n, float)
    np.testing.assert_array_equal(en, [1.2, 1.2, 1.2])

    # --- continuous numpy
    assert space.shape == (3,)
    np.testing.assert_array_equal(space.low, [-1, -1, -1])
    np.testing.assert_array_equal(space.high, [3, 3, 3])
    de = space.decode_from_np(np.array([1.2, 1.2, 1.2]))
    assert isinstance(de, list)
    for n in de:
        assert isinstance(n, float)
    np.testing.assert_array_equal(de, [1.2, 1.2, 1.2])
    en = space.encode_to_np([1.2, 1.2, 1.2], np.float32)
    np.testing.assert_array_equal(en, np.array([1.2, 1.2, 1.2], dtype=np.float32))

    # --- sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, list)
        assert len(action) == 3
        for a in action:
            assert isinstance(a, float)
        assert -1 <= action[0] <= 3
        assert -1 <= action[1] <= 3
        assert -1 <= action[2] <= 3

    # --- eq
    assert space == ArrayContinuousSpace(3, -1, 3)
    assert space != ArrayContinuousSpace(3, -1, 2)


def test_discrete_no_division():
    space = ArrayContinuousSpace(3, -1, 3)

    # --- discrete
    with pytest.raises(AssertionError):
        _ = space.n
    with pytest.raises(AssertionError):
        _ = space.encode_to_int([1.2, 1.2, 1.2])
    de = space.decode_from_int(2)
    np.testing.assert_array_equal(de, [2.0, 2.0, 2.0])

    # --- discrete numpy
    en = space.encode_to_list_int([1.2, 1.2, 1.2])
    assert len(en) == 3
    np.testing.assert_array_equal(en, [1, 1, 1])
    de = space.decode_from_list_int([1, 1, 1])
    assert isinstance(de, list)
    for n in de:
        assert isinstance(n, float)
    np.testing.assert_array_equal(de, [1.0, 1.0, 1.0])


def test_discrete_division():
    space = ArrayContinuousSpace(3, -1, 3)

    # action discrete
    space.create_division_tbl(5)
    assert space.division_tbl is not None

    # --- discrete
    assert space.n == 5**3
    en = space.encode_to_int([-1, -1, 2])
    assert en == 3
    de = space.decode_from_int(2)
    assert isinstance(de, list)
    for n in de:
        assert isinstance(n, float)
    np.testing.assert_array_equal(de, [-1.0, -1.0, 1.0])

    # --- discrete numpy
    en = space.encode_to_list_int([-1, -1, 2])
    assert len(en) == 1
    assert en[0] == 3
    de = space.decode_from_list_int([2])
    assert isinstance(de, list)
    for n in de:
        assert isinstance(n, float)
    np.testing.assert_array_equal(de, [-1.0, -1.0, 1.0])


def test_inf():
    space = ArrayContinuousSpace(3)

    # sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, list)
        assert len(action) == 3
        for a in action:
            assert isinstance(a, float)

    # --- discrete
    space.create_division_tbl(5)
    with pytest.raises(AssertionError):
        _ = space.encode_to_int([1.2, 1.2, 1.2])

    # --- continuous list
    assert space.list_size == 3
    np.testing.assert_array_equal(space.list_low, [-np.inf, -np.inf, -np.inf])
    np.testing.assert_array_equal(space.list_high, [np.inf, np.inf, np.inf])


def test_sanitize():
    space = ArrayContinuousSpace(3, -1, 3)

    val = space.sanitize(1)
    assert space.check_val(val)
    np.testing.assert_array_equal([1.0, 1.0, 1.0], val)

    val = space.sanitize([1.2, 0.9, 0.8])
    assert space.check_val(val)
    np.testing.assert_array_equal([1.2, 0.9, 0.8], val)

    val = space.sanitize((2, 1, True))
    assert space.check_val(val)
    np.testing.assert_array_equal([2.0, 1.0, 1.0], val)

    val = space.sanitize(np.array([1.2, 0.9, 0.8]))
    assert space.check_val(val)
    np.testing.assert_array_equal([1.2, 0.9, 0.8], val)

    assert not space.check_val(1)
    assert not space.check_val([1])
    assert not space.check_val([1.1, 1.1, 1])
    assert not space.check_val([-2.1, 1.1, 1.1])
    assert not space.check_val([5.1, 1.1, 1.1])


def test_sanitize2():
    space = ArrayContinuousSpace(3, -1, 3)

    val = space.sanitize([-2, 6, 2])
    assert space.check_val(val)
    np.testing.assert_array_equal([-1, 3, 2], val)
