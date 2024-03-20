import itertools

import numpy as np
import pytest

from srl.base.define import SpaceTypes
from srl.base.spaces import BoxSpace


def test_space_basic():
    space = BoxSpace((3, 1), -1, 3)

    assert space.shape == (3, 1)
    np.testing.assert_array_equal(space.low, np.full((3, 1), -1))
    np.testing.assert_array_equal(space.high, np.full((3, 1), 3))
    assert space.stype == SpaceTypes.CONTINUOUS
    assert space.dtype == np.float32

    # --- check_val
    assert space.check_val(np.full((3, 1), 0))
    assert not space.check_val(np.full((3, 1), -2))

    # --- to_str
    assert space.to_str(np.full((3, 1), 0)) == "0,0,0"

    # --- copy
    c = space.copy()
    assert space == c

    # --- get_default
    assert (space.get_default() == np.zeros((3, 1))).all()

    # --- stack
    o = space.create_stack_space(3)
    assert isinstance(o, BoxSpace)
    assert o == BoxSpace((3, 3, 1), -1, 3)


def test_space_type():
    space = BoxSpace((2, 3), 0, 3, dtype=np.int8)
    assert space.stype == SpaceTypes.DISCRETE
    assert space.dtype == np.int8
    assert space != BoxSpace((2, 3), 0, 3, dtype=np.uint8)

    space = BoxSpace((2, 3), -1, 3, stype=SpaceTypes.COLOR)
    assert space.stype == SpaceTypes.COLOR
    assert space.dtype == np.float32
    assert space != BoxSpace((2, 3), -1, 3, stype=SpaceTypes.GRAY_3ch)


def test_discrete():
    space = BoxSpace((3, 1), 0, [[2], [5], [3]], np.int64)
    print(space)
    assert space.stype == SpaceTypes.DISCRETE

    # --- discrete
    assert space.int_size == (2 + 1) * (5 + 1) * (3 + 1)
    de = space.decode_from_int(1)
    assert isinstance(de, np.ndarray)
    assert de.shape == (3, 1)
    np.testing.assert_array_equal(de, [[0], [0], [1]])
    en = space.encode_to_int(np.array([[0], [0], [1]], np.int64))
    assert isinstance(en, int)
    assert en == 1

    # --- list int
    assert space.list_int_size == 3
    assert space.list_int_low == [0, 0, 0]
    assert space.list_int_high == [2, 5, 3]
    de = space.decode_from_list_int([1, 2, 0])
    assert isinstance(de, np.ndarray)
    assert de.shape == (3, 1)
    np.testing.assert_array_equal(de, [[1], [2], [0]])
    en = space.encode_to_list_int(np.array([[1], [2], [0]], np.int64))
    assert isinstance(en, list)
    assert len(en) == 3
    np.testing.assert_array_equal(en, [1, 2, 0])

    # --- list float
    assert space.list_float_size == 3
    np.testing.assert_array_equal(space.list_float_low, [0, 0, 0])
    np.testing.assert_array_equal(space.list_float_high, [2, 5, 3])
    de = space.decode_from_list_float([0.1, 0.6, 1.9])
    assert isinstance(de, np.ndarray)
    assert de.shape == (3, 1)
    np.testing.assert_array_equal(de, [[0], [0], [1]])
    en = space.encode_to_list_float(np.array([[0], [1], [1]], np.int64))
    assert isinstance(en, list)
    for n in en:
        assert isinstance(n, float)
    np.testing.assert_array_equal(en, [0.0, 1.0, 1.0])

    # --- continuous numpy
    assert space.shape == (3, 1)
    np.testing.assert_array_equal(space.low, [[0], [0], [0]])
    np.testing.assert_array_equal(space.high, [[2], [5], [3]])
    de = space.decode_from_np(np.array([[0.1], [0.6], [1.9]]))
    assert isinstance(de, np.ndarray)
    assert de.shape == (3, 1)
    np.testing.assert_array_equal(de, [[0], [0], [1]])
    en = space.encode_to_np(np.array([[0], [1], [1]], np.int64), dtype=np.float32)
    np.testing.assert_array_equal(en, np.array([[0], [1], [1]], np.float32))

    # --- sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, np.ndarray)
        assert action.shape == (3, 1)
        assert 0 <= action[0][0] <= 2
        assert 0 <= action[1][0] <= 5
        assert 0 <= action[2][0] <= 3
        print(action)

    # --- eq
    assert space == BoxSpace((3, 1), 0, [[2], [5], [3]], np.int64)
    assert space != BoxSpace((3, 1), 0, [[3], [5], [3]], np.int64)


def test_con_no_division():
    space = BoxSpace((3, 2), -1, 3)
    print(space)

    # --- discrete
    with pytest.raises(AssertionError):
        _ = space.int_size
    with pytest.raises(AssertionError):
        _ = space.encode_to_int(np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]]))
    de = space.decode_from_int(2)
    assert de.shape == (3, 2)
    np.testing.assert_array_equal(de, np.array([[2, 2], [2, 2], [2, 2]], np.float32))

    # --- list int
    assert space.list_int_size == 6
    assert space.list_int_low == [-1] * 6
    assert space.list_int_high == [3] * 6
    en = space.encode_to_list_int(np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]]))
    assert len(en) == 6
    assert en == [1, 2, 3, -1, 2, 2]
    de = space.decode_from_list_int([1, 2, 3, -1, 1, 1])
    assert de.shape == (3, 2)
    np.testing.assert_array_equal(de, np.array([[1, 2], [3, -1], [1, 1]], np.float32))

    # --- list float
    assert space.list_float_size == 6
    np.testing.assert_array_equal(space.list_float_low, [-1] * 6)
    np.testing.assert_array_equal(space.list_float_high, [3] * 6)
    de = space.decode_from_list_float([1.1, 2.2, 3.0, -1.0, 1.5, 1.6])
    np.testing.assert_array_equal(de, np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]], np.float32))
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
    np.testing.assert_array_equal(de, np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]], np.float32))
    en = space.encode_to_np(np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]]), np.float32)
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


def test_con_division():
    space = BoxSpace((3, 2), -1, 3)

    # action discrete
    space.create_division_tbl(5)
    assert space.division_tbl is not None
    _t = list(itertools.product([-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]))
    true_tbl = list(itertools.product(_t, _t, _t))
    for a in range(len(true_tbl)):
        np.testing.assert_array_equal(true_tbl[a], space.division_tbl[a])

    # --- discrete
    assert space.int_size == 5 ** (3 * 2)
    en = space.encode_to_int(np.array([[-0.9, -1], [-1, -0.9], [-1, -1]]))
    assert en == 0
    de = space.decode_from_int(0)
    np.testing.assert_array_equal(de, [[-1, -1], [-1, -1], [-1, -1]])

    # --- list int
    en = space.encode_to_list_int(np.array([[-0.9, -1], [-1, -0.9], [-1, -1]]))
    assert len(en) == 1
    assert en[0] == 0
    de = space.decode_from_list_int([0])
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

    # --- list float
    assert space.list_float_size == 6
    np.testing.assert_array_equal(space.list_float_low, [-np.inf] * 6)
    np.testing.assert_array_equal(space.list_float_high, [np.inf] * 6)


def test_sanitize():
    space = BoxSpace((2, 1), -1, 3)

    val = space.sanitize([[1.2], [2]])
    assert space.check_val(val)
    np.testing.assert_array_equal(np.array([[1.2], [2.0]], dtype=np.float32), val)

    assert not space.check_val(np.array([1.2, 2.2], dtype=np.float32))
    assert not space.check_val([[1.2], [2.2]])
    assert not space.check_val(np.array([[1.2], [2.2], [2.2]], dtype=np.float32))
    assert not space.check_val(np.array([[1.2], [-1.1]], dtype=np.float32))
    assert not space.check_val(np.array([[1.2], [3.1]], dtype=np.float32))


def test_sanitize2():
    space = BoxSpace((2, 1), -1, 3)

    val = space.sanitize([[-2], [6]])
    assert space.check_val(val)
    np.testing.assert_array_equal([[-1], [3]], val)


def test_sample1():
    space = BoxSpace((1, 1), -1, 0, np.int64)
    for _ in range(100):
        r = space.sample().tolist()[0][0]
        assert isinstance(r, int)
        assert r in [-1, 0]


def test_sample2():
    space = BoxSpace((2, 1), -1, 0, np.int64)
    for _ in range(100):
        r = space.sample(
            [
                np.array([[0], [0]], np.int64),
                np.array([[-1], [0]], np.int64),
                np.array([[0], [-1]], np.int64),
            ]
        )
        assert r.tolist() == [[-1], [-1]]


def test_valid_actions():
    space = BoxSpace((2, 1), -1, 0, np.int64)
    acts = space.get_valid_actions(
        [
            np.array([[-1], [-1]], np.int64),
            np.array([[-1], [0]], np.int64),
            np.array([[0], [-1]], np.int64),
        ]
    )
    assert len(acts) == 1
    assert (acts[0] == [[0], [0]]).all()
