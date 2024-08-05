import numpy as np
import pytest

from srl.base import spaces
from srl.base.define import SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.continuous import ContinuousSpace


def test_space_basic():
    space = ContinuousSpace(-1, 3)

    assert space.low == -1
    assert space.high == 3
    assert space.stype == SpaceTypes.CONTINUOUS
    assert space.dtype == np.float32

    # --- check_val
    assert space.check_val(1.2)
    assert not space.check_val(-2)

    # --- to_str
    assert space.to_str(1.2) == "1.2"
    assert space.to_str(1.0) == "1"

    # --- copy
    c = space.copy()
    assert space == c

    # --- stack
    o = space.create_stack_space(3)
    assert isinstance(o, ArrayContinuousSpace)
    assert o == ArrayContinuousSpace(3, -1, 3)
    v = space.encode_stack([1, 1, 0])
    assert v == [1, 1, 0]


def test_space_get_default():
    space = ContinuousSpace(-1, 3)
    assert space.get_default() == 0.0
    space = ContinuousSpace(2, 3)
    assert space.get_default() == 2.0
    space = ContinuousSpace(-2, -1)
    assert space.get_default() == -2.0


def test_no_division():
    space = ContinuousSpace(-1.8, 3.1)
    print(space)

    # --- discrete
    with pytest.raises(AssertionError):
        _ = space.int_size
    en = space.encode_to_int(1.2)
    assert en == 1
    de = space.decode_from_int(2)
    assert de == 2.0

    # --- list int
    assert space.list_int_size == 1
    assert space.list_int_low == [-2]
    assert space.list_int_high == [3]
    en = space.encode_to_list_int(1.2)
    assert len(en) == 1
    assert en[0] == 1
    de = space.decode_from_list_int([1])
    assert de == 1.0

    # --- list float
    assert space.list_float_size == 1
    np.testing.assert_array_equal(space.list_float_low, [-1.8])
    np.testing.assert_array_equal(space.list_float_high, [3.1])
    de = space.decode_from_list_float([1.2])
    assert isinstance(de, float)
    assert de == 1.2
    en = space.encode_to_list_float(1.2)
    assert isinstance(en, list)
    assert en == [1.2]

    # --- continuous numpy
    assert space.np_shape == (1,)
    np.testing.assert_array_equal(space.low, [-1.8])
    np.testing.assert_array_equal(space.high, [3.1])
    de = space.decode_from_np(np.array([1.2]))
    assert isinstance(de, float)
    assert de == 1.2
    en = space.encode_to_np(1.2, np.float32)
    np.testing.assert_array_equal(en, np.array([1.2], dtype=np.float32))

    # --- sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, float)
        assert action >= -1.8
        assert action <= 3.1

    # --- eq
    assert space == ContinuousSpace(-1.8, 3.1)
    assert space != ContinuousSpace(-1, 2)


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
    assert space.int_size == 5
    en = space.encode_to_int(1.2)
    assert en == 2
    de = space.decode_from_int(2)
    assert de == 1.0

    # --- list int
    assert space.list_int_size == 1
    assert space.list_int_low == [0]
    assert space.list_int_high == [5]
    en = space.encode_to_list_int(1.2)
    assert en == [2]
    de = space.decode_from_list_int([1])
    assert de == 0.0


def test_inf():
    space = ContinuousSpace()

    print(space)

    # sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, float)

    # --- discrete
    space.create_division_tbl(5)
    en = space.encode_to_int(1.2)
    assert en == 1

    # --- continuous list
    assert space.list_float_size == 1
    np.testing.assert_array_equal(space.list_float_low, [-np.inf])
    np.testing.assert_array_equal(space.list_float_high, [np.inf])


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


@pytest.mark.parametrize(
    "create_space, true_space, val, decode_val",
    [
        ["", ContinuousSpace(-1, 3), 1.1, 1.1],
        ["DiscreteSpace", spaces.DiscreteSpace(5, 0), 1, 0],
        ["ArrayDiscreteSpace", spaces.ArrayDiscreteSpace(1, -1, 3), [2], 2.0],
        ["ContinuousSpace", spaces.ContinuousSpace(-1, 3), 1.1, 1.1],
        ["ArrayContinuousSpace", spaces.ArrayContinuousSpace(1, -1, 3), [1.1], 1.1],
        ["BoxSpace", spaces.BoxSpace((1,), -1, 3), np.full((1,), 0.25, np.float32), 0.25],
        ["BoxSpace_float", spaces.BoxSpace((1,), -1, 3), np.full((1,), 0.25, np.float32), 0.25],
        ["TextSpace", None, "2", 3],
    ],
)
def test_space(create_space, true_space, val, decode_val):
    space = ContinuousSpace(-1, 3)
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
