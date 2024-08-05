import numpy as np
import pytest

from srl.base import spaces
from srl.base.define import SpaceTypes
from srl.base.spaces.text import TextSpace


def test_space_basic():
    space = TextSpace(5)

    assert space.stype == SpaceTypes.DISCRETE
    assert space.dtype == np.uint64

    # --- check_val
    assert space.check_val("a")
    assert not space.check_val(0)

    # --- to_str
    assert space.to_str("a") == "a"

    # --- get_default
    assert space.get_default() == "aaaaa"

    # --- copy
    c = space.copy()
    assert space == c

    # --- stack
    o = space.create_stack_space(3)
    assert isinstance(o, TextSpace)
    assert o == TextSpace(5 * 3)
    v = space.encode_stack(["a", "b", "c"])
    assert v == "abc"


def test_space_encode():
    space = TextSpace(3)
    print(space)

    # --- discrete
    with pytest.raises(NotImplementedError):
        assert space.int_size == 5
        de = space.decode_from_int(2)
        assert isinstance(de, int)
        assert de == 3
        en = space.encode_to_int(3)
        assert isinstance(en, int)
        assert en == 2

    # --- list int
    assert space.list_int_size == 3
    assert space.list_int_low == [0, 0, 0]
    assert space.list_int_high == [0x7F, 0x7F, 0x7F]
    en = space.encode_to_list_int("ab")
    assert isinstance(en, list)
    assert len(en) == 3
    assert en[0] == ord("a")
    assert en[1] == ord("b")
    assert en[2] == ord(" ")
    de = space.decode_from_list_int(en)
    assert de == "ab "

    # --- list float
    assert space.list_float_size == 3
    np.testing.assert_array_equal(space.list_float_low, [0, 0, 0])
    np.testing.assert_array_equal(space.list_float_high, [0x7F, 0x7F, 0x7F])
    en = space.encode_to_list_float("ab")
    assert len(en) == 3
    assert en[0] == ord("a")
    assert en[1] == ord("b")
    assert en[2] == ord(" ")
    de = space.decode_from_list_float(en)
    assert de == "ab "

    # --- continuous numpy
    assert space.np_shape == (3,)
    np.testing.assert_array_equal(space.np_low, [0, 0, 0])
    np.testing.assert_array_equal(space.np_high, [0x7F, 0x7F, 0x7F])
    en = space.encode_to_np("ab", np.float32)
    assert isinstance(en, np.ndarray)
    assert len(en) == 3
    assert en[0] == ord("a")
    assert en[1] == ord("b")
    assert en[2] == ord(" ")
    de = space.decode_from_np(en)
    assert de == "ab "

    # --- sample
    actions = [space.sample() for _ in range(100)]
    print(actions)

    # --- eq
    assert space == TextSpace(3)
    assert space != TextSpace(4)
    assert space != "a"


def test_sanitize():
    space = TextSpace(3)

    val = space.sanitize("ab")
    assert space.check_val(val)
    assert val == "ab "

    assert not space.check_val(1)
    assert not space.check_val("abcd")


@pytest.mark.parametrize(
    "create_space, true_space, val, decode_val",
    [
        ["", spaces.DiscreteSpace(5, 0), 2, 3],
        ["DiscreteSpace", spaces.DiscreteSpace(5, 0), 2, 3],
        ["ArrayDiscreteSpace", spaces.ArrayDiscreteSpace(1, 0, 4), [2], 3],
        ["ContinuousSpace", spaces.ContinuousSpace(0, 4), 2, 3],
        ["ArrayContinuousSpace", spaces.ArrayContinuousSpace(1, 0, 4), [2], 3],
        ["BoxSpace", spaces.BoxSpace((1,), 0, 4, np.int64), np.full((1,), 2), 3],
        ["BoxSpace_float", spaces.BoxSpace((1,), 0, 4, np.float32), np.full((1,), 2), 3],
        ["TextSpace", None, "2", 3],
    ],
)
def test_space(create_space, true_space, val, decode_val):
    pytest.skip("TODO")
