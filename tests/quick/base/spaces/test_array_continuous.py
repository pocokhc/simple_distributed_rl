import numpy as np
import pytest

from srl.base import spaces
from srl.base.define import SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.array_continuous import ArrayContinuousSpace


def test_space_basic():
    space = ArrayContinuousSpace(3, -1.8, 3.1)

    assert space.size == 3
    np.testing.assert_array_equal(space.low, np.array([-1.8, -1.8, -1.8], np.float32))
    np.testing.assert_array_equal(space.high, np.array([3.1, 3.1, 3.1], np.float32))
    assert space.stype == SpaceTypes.CONTINUOUS
    assert space.dtype == np.float32

    # --- check_val
    assert space.check_val([0.0, 0.0, 0.0])
    assert not space.check_val([0.0, -2.0, 0.0])
    assert not space.check_val([0.0, 0.0])

    # --- to_str
    assert space.to_str([0.1, 0.5, 1.0]) == "0.1,0.5,1"

    # --- copy
    c = space.copy()
    assert space == c

    # --- stack
    o = space.create_stack_space(3)
    assert isinstance(o, ArrayContinuousSpace)
    assert o == ArrayContinuousSpace(3 * 3, -1.8, 3.1)
    v = space.encode_stack([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    assert v == [1, 1, 0, 1, 1, 0, 1, 1, 0]


def test_space_get_default():
    space = ArrayContinuousSpace(3, -1.8, 3.1)
    assert space.get_default() == [0, 0, 0]
    space = ArrayContinuousSpace(3, 1, [2, 5, 3])
    assert space.get_default() == [1, 1, 1]
    space = ArrayContinuousSpace(3, -2, -1)
    assert space.get_default() == [-2, -2, -2]


def test_no_division():
    space = ArrayContinuousSpace(3, -1.8, 3.1)
    print(space)

    # --- int
    with pytest.raises(AssertionError):
        _ = space.int_size
    with pytest.raises(AssertionError):
        _ = space.encode_to_int([1.2, 1.2, 1.2])
    de = space.decode_from_int(2)
    assert de == [2.0, 2.0, 2.0]

    # --- list int
    assert space.list_int_size == 3
    np.testing.assert_array_equal(space.list_int_low, np.array([-2, -2, -2], np.float32))
    np.testing.assert_array_equal(space.list_int_high, np.array([3, 3, 3], np.float32))
    en = space.encode_to_list_int([1.2, 1.2, 1.2])
    assert en == [1, 1, 1]
    de = space.decode_from_list_int([1, 1, 1])
    assert de == [1.0, 1.0, 1.0]

    # --- list float
    assert space.list_float_size == 3
    np.testing.assert_array_equal(space.list_float_low, np.array([-1.8, -1.8, -1.8], np.float32))
    np.testing.assert_array_equal(space.list_float_high, np.array([3.1, 3.1, 3.1], np.float32))
    de = space.decode_from_list_float([1.2, 1.2, 1.2])
    assert de == [1.2, 1.2, 1.2]
    en = space.encode_to_list_float([1.2, 1.2, 1.2])
    assert en == [1.2, 1.2, 1.2]

    # --- continuous numpy
    assert space.np_shape == (3,)
    np.testing.assert_array_equal(space.np_low, np.array([-1.8, -1.8, -1.8], np.float32))
    np.testing.assert_array_equal(space.np_high, np.array([3.1, 3.1, 3.1], np.float32))
    de = space.decode_from_np(np.array([1.2, 1.2, 1.2]))
    assert isinstance(de, list)
    for n in de:
        assert isinstance(n, float)
    np.testing.assert_array_equal(de, [1.2, 1.2, 1.2])
    en = space.encode_to_np([1.2, 1.2, 1.2], np.float32)
    assert isinstance(en, np.ndarray)
    np.testing.assert_array_equal(en, np.array([1.2, 1.2, 1.2], dtype=np.float32))

    # --- sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, list)
        assert len(action) == 3
        for a in action:
            assert isinstance(a, float)
        assert -1.8 <= action[0] <= 3.1
        assert -1.8 <= action[1] <= 3.1
        assert -1.8 <= action[2] <= 3.1

    # --- eq
    assert space == ArrayContinuousSpace(3, -1.8, 3.1)
    assert space != ArrayContinuousSpace(3, -1.8, 3.0)


def test_division():
    space = ArrayContinuousSpace(3, -1, 3)

    # action discrete
    space.create_division_tbl(5)
    assert space.division_tbl is not None
    print(space.division_tbl)

    # --- discrete
    assert space.int_size == 8
    en = space.encode_to_int([-1, -1, 2])
    assert en == 1
    de = space.decode_from_int(2)
    assert de == [-1.0, 3.0, -1.0]

    # --- list int
    assert space.list_int_size == 1
    assert space.list_int_low == [0]
    assert space.list_int_high == [8]
    en = space.encode_to_list_int([-1, -1, 2])
    assert en == [1]
    de = space.decode_from_list_int([2])
    assert de == [-1.0, 3.0, -1.0]


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
    assert space.list_float_size == 3
    np.testing.assert_array_equal(space.list_float_low, [-np.inf, -np.inf, -np.inf])
    np.testing.assert_array_equal(space.list_float_high, [np.inf, np.inf, np.inf])


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


@pytest.mark.parametrize(
    "create_space, true_space, val, decode_val",
    [
        ["", spaces.ArrayContinuousSpace(2, -1, 3), [1.1, 1.0], [1.1, 1.0]],
        ["DiscreteSpace", spaces.DiscreteSpace(4, 0), 2, [3.0, -1.0]],
        ["ArrayDiscreteSpace", spaces.ArrayDiscreteSpace(2, -1, 3), [1, 1], [1.0, 1.0]],
        ["ContinuousSpace", None, 1.0, [1.0, 1.0]],
        ["ArrayContinuousSpace", spaces.ArrayContinuousSpace(2, -1, 3), [1.1, 1.0], [1.1, 1.0]],
        ["BoxSpace", spaces.BoxSpace((2,), -1, 3), np.full((2,), 2.0), [2.0, 2.0]],
        ["BoxSpace_float", spaces.BoxSpace((2,), -1, 3), np.full((2,), 2.0), [2.0, 2.0]],
        ["TextSpace", None, "2", 3],
    ],
)
def test_space(create_space, true_space, val, decode_val):
    space = ArrayContinuousSpace(2, -1, 3)
    print(space)

    if true_space is None:
        with pytest.raises(NotSupportedError):
            space.create_encode_space(create_space)
        return

    if create_space in ["DiscreteSpace"]:
        space.create_division_tbl(5)
    target_space = space.create_encode_space(create_space)
    print(target_space)
    assert target_space == true_space

    de = space.decode_from_space(val, target_space)
    print(de)
    if isinstance(de, np.ndarray):
        assert (de == decode_val).all()
    else:
        assert de == decode_val
    assert space.check_val(de)
    en = space.encode_to_space(decode_val, target_space)
    if isinstance(en, np.ndarray):
        assert (en == val).all()
    else:
        assert en == val
    assert target_space.check_val(en)

    de = space.decode_from_space(en, target_space)
    if isinstance(de, np.ndarray):
        assert (de == decode_val).all()
    else:
        assert de == decode_val
    assert space.check_val(de)
