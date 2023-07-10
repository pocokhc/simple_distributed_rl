import itertools

import numpy as np
import pytest

from srl.base.define import RLTypes
from srl.base.spaces import BoxSpace


def test_1dim():
    space = BoxSpace((1,), -1, 3)
    assert space.rl_type == RLTypes.CONTINUOUS

    print(space)

    # --- continuous list
    assert space.list_size == 1
    np.testing.assert_array_equal(space.list_low, [-1])
    np.testing.assert_array_equal(space.list_high, [3])
    de = space.decode_from_list_float([1.1])
    np.testing.assert_array_equal(de, [1.1])
    en = space.encode_to_list_float(np.array([1.2]))
    assert isinstance(en, list)
    for n in en:
        assert isinstance(n, float)
    assert en[0] == 1.2

    # --- continuous numpy
    assert space.shape == (1,)
    np.testing.assert_array_equal(space.low, [-1])
    np.testing.assert_array_equal(space.high, [3])
    de = space.decode_from_np(np.array([1.1]))
    np.testing.assert_array_equal(de, [1.1])
    en = space.encode_to_np(np.array([1.2]))
    np.testing.assert_array_equal(en, np.array([1.2], dtype=np.float32))

    # --- sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, np.ndarray)
        assert "float" in str(action.dtype)
        assert action.shape == (1,)
        assert np.min(action) >= -1
        assert np.max(action) <= 3

    # --- eq
    assert space == BoxSpace((1,), -1, 3)
    assert space != BoxSpace((1,), -1, 2)


def test_2dim():
    space = BoxSpace((3, 2), -1, 3)
    assert space.rl_type == RLTypes.CONTINUOUS

    print(space)

    # --- continuous list
    assert space.list_size == 6
    np.testing.assert_array_equal(space.list_low, [-1] * 6)
    np.testing.assert_array_equal(space.list_high, [3] * 6)
    de = space.decode_from_list_float([1.1, 2.2, 3.0, -1.0, 1.5, 1.6])
    np.testing.assert_array_equal(de, [[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]])
    en = space.encode_to_list_float(np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]]))
    assert isinstance(en, list)
    for n in en:
        assert isinstance(n, float)
    np.testing.assert_array_equal(en, [1.1, 2.2, 3.0, -1.0, 1.5, 1.6])

    # --- continuous numpy
    assert space.shape == (3, 2)
    np.testing.assert_array_equal(space.low, np.array([-1] * 6).reshape((3, 2)))
    np.testing.assert_array_equal(space.high, np.array([3] * 6).reshape((3, 2)))
    de = space.decode_from_np(np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]]))
    np.testing.assert_array_equal(de, [[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]])
    en = space.encode_to_np(np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]]))
    np.testing.assert_array_equal(en, np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]], dtype=np.float32))

    # --- sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, np.ndarray)
        assert "float" in str(action.dtype)
        assert action.shape == (3, 2)
        assert np.min(action) >= -1
        assert np.max(action) <= 3

    # --- eq
    assert space == BoxSpace((3, 2), -1, 3)
    assert space != BoxSpace((3, 2), -1, 2)
    assert space != BoxSpace((3, 3), -1, 3)


def test_discrete_no_division():
    space = BoxSpace((3, 2), -1, 3)

    # --- discrete
    with pytest.raises(AssertionError):
        _ = space.n
    with pytest.raises(AssertionError):
        _ = space.encode_to_int(np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]]))
    de = space.decode_from_int(2)
    assert de.shape == (3, 2)
    np.testing.assert_array_equal(de, [[2, 2], [2, 2], [2, 2]])

    # --- discrete numpy
    en = space.encode_to_int_np(np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]]))
    assert en.shape == (3, 2)
    np.testing.assert_array_equal(en, [[1, 2], [3, -1], [2, 2]])
    de = space.decode_from_int_np(np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]]))
    assert de.shape == (3, 2)
    np.testing.assert_array_equal(de, [[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]])


def test_discrete_division():
    space = BoxSpace((3, 2), -1, 3)

    # action discrete
    space.create_division_tbl(5)
    assert space.division_tbl is not None
    _t = list(itertools.product([-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]))
    true_tbl = list(itertools.product(_t, _t, _t))
    for a in range(len(true_tbl)):
        np.testing.assert_array_equal(true_tbl[a], space.division_tbl[a])

    # --- discrete
    assert space.n == 5 ** (3 * 2)
    en = space.encode_to_int(np.array([[-0.9, -1], [-1, -0.9], [-1, -1]]))
    assert en == 0
    de = space.decode_from_int(0)
    np.testing.assert_array_equal(de, [[-1, -1], [-1, -1], [-1, -1]])

    # --- discrete numpy
    en = space.encode_to_int_np(np.array([[-0.9, -1], [-1, -0.9], [-1, -1]]))
    assert en.shape == (1,)
    assert en[0] == 0
    de = space.decode_from_int_np(np.array([0]))
    np.testing.assert_array_equal(de, [[-1, -1], [-1, -1], [-1, -1]])


def test_inf():
    space = BoxSpace((3, 2))

    # sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, np.ndarray)
        assert "float" in str(action.dtype)
        assert action.shape == (3, 2)

    # --- discrete
    space.create_division_tbl(5)
    with pytest.raises(AssertionError):
        _ = space.encode_to_int(np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]]))

    # --- continuous list
    assert space.list_size == 6
    np.testing.assert_array_equal(space.list_low, [-np.inf] * 6)
    np.testing.assert_array_equal(space.list_high, [np.inf] * 6)


def test_convert():
    space = BoxSpace((2, 1), -1, 3)

    val = space.convert([[1.2], [2]])
    assert space.check_val(val)
    np.testing.assert_array_equal(np.array([[1.2], [2.0]], dtype=np.float32), val)

    assert not space.check_val(np.array([1.2, 2.2], dtype=np.float32))
    assert not space.check_val([[1.2], [2.2]])
    assert not space.check_val(np.array([[1.2], [2.2], [2.2]], dtype=np.float32))
    assert not space.check_val(np.array([[1.2], [-1.1]], dtype=np.float32))
    assert not space.check_val(np.array([[1.2], [3.1]], dtype=np.float32))


def test_convert2():
    space = BoxSpace((2, 1), -1, 3)

    val = space.convert([[-2], [6]])
    assert space.check_val(val)
    np.testing.assert_array_equal([[-1], [3]], val)
