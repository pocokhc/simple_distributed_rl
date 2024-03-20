import numpy as np

from srl.base.define import SpaceTypes
from srl.base.spaces import ArrayDiscreteSpace


def test_space_basic():
    space = ArrayDiscreteSpace(3, 0, [2, 5, 3])

    assert space.size == 3
    assert space.low == [0, 0, 0]
    assert space.high == [2, 5, 3]
    assert space.stype == SpaceTypes.DISCRETE
    assert space.dtype == np.int64

    # --- check_val
    assert space.check_val([0, 0, 0])
    assert not space.check_val([0, -1, 0])
    assert not space.check_val([0, 0])

    # --- to_str
    assert space.to_str([0, 0, 0]) == "0,0,0"

    # --- copy
    c = space.copy()
    assert space == c

    # --- get_default
    assert space.get_default() == [0, 0, 0]
    space = ArrayDiscreteSpace(3, 1, [2, 5, 3])
    assert space.get_default() == [1, 1, 1]
    space = ArrayDiscreteSpace(3, -2, -1)
    assert space.get_default() == [-2, -2, -2]

    # --- stack
    o = space.create_stack_space(3)
    assert isinstance(o, ArrayDiscreteSpace)
    assert o == ArrayDiscreteSpace(3 * 3, 0, [2, 5, 3] * 3)


def test_space():
    space = ArrayDiscreteSpace(3, 0, [2, 5, 3])
    print(space)

    # --- discrete
    assert space.int_size == (2 + 1) * (5 + 1) * (3 + 1)
    de = space.decode_from_int(1)
    assert isinstance(de, list)
    for n in de:
        assert isinstance(n, int)
    np.testing.assert_array_equal(de, [0, 0, 1])
    en = space.encode_to_int([0, 0, 1])
    assert isinstance(en, int)
    assert en == 1

    # --- list int
    assert space.list_int_size == 3
    assert space.list_int_low == [0, 0, 0]
    assert space.list_int_high == [2, 5, 3]
    de = space.decode_from_list_int([1, 2, 0])
    assert de == [1, 2, 0]
    en = space.encode_to_list_int([1, 2, 0])
    assert en == [1, 2, 0]

    # --- list float
    assert space.list_float_size == 3
    assert space.list_int_low == [0, 0, 0]
    assert space.list_int_high == [2, 5, 3]
    de = space.decode_from_list_float([0.1, 0.6, 0.9])
    assert de == [0, 1, 1]
    en = space.encode_to_list_float([0, 1, 1])
    assert en == [0.0, 1.0, 1.0]

    # --- continuous numpy
    assert space.np_shape == (3,)
    np.testing.assert_array_equal(space.low, [0, 0, 0])
    np.testing.assert_array_equal(space.high, [2, 5, 3])
    de = space.decode_from_np(np.array([0.1, 0.6, 0.9]))
    assert de == [0, 1, 1]
    en = space.encode_to_np([0, 1, 1], dtype=np.float32)
    assert isinstance(en, np.ndarray)
    np.testing.assert_array_equal(en, [0, 1, 1])

    # --- sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, list)
        assert len(action) == 3
        for a in action:
            assert isinstance(a, int)
        assert 0 <= action[0] <= 2
        assert 0 <= action[1] <= 5
        assert 0 <= action[2] <= 3

    # --- eq
    assert space == ArrayDiscreteSpace(3, 0, [2, 5, 3])
    assert space != ArrayDiscreteSpace(3, 0, [3, 5, 3])


def test_sanitize():
    space = ArrayDiscreteSpace(3, 0, [2, 5, 3])

    val = space.sanitize(1.2)
    assert space.check_val(val)
    np.testing.assert_array_equal([1, 1, 1], val)

    val = space.sanitize([1.2, 0.9, 1.1])
    assert space.check_val(val)
    np.testing.assert_array_equal([1, 1, 1], val)

    val = space.sanitize((0, 4, True))
    assert space.check_val(val)
    np.testing.assert_array_equal([0, 4, 1], val)

    val = space.sanitize(np.array([1.2, 0.9, 1.1]))
    assert space.check_val(val)
    np.testing.assert_array_equal([1, 1, 1], val)

    assert not space.check_val(1)
    assert not space.check_val([1, 2])
    assert not space.check_val([1, 1, 1.1])
    assert not space.check_val([-1, 1, 1])
    assert not space.check_val([2, 5, 4])


def test_sanitize2():
    space = ArrayDiscreteSpace(3, 0, [2, 5, 3])

    val = space.sanitize([-1, 6, 2])
    assert space.check_val(val)
    np.testing.assert_array_equal([0, 5, 2], val)


def test_sample():
    space = ArrayDiscreteSpace(2, 0, 1)
    for _ in range(100):
        r = space.sample(
            [
                [0, 0],
                [1, 0],
                [0, 1],
            ]
        )
        assert r == [1, 1]


def test_valid_actions():
    space = ArrayDiscreteSpace(2, 0, 1)
    acts = space.get_valid_actions(
        [
            [0, 0],
            [1, 0],
            [0, 1],
        ]
    )
    assert len(acts) == 1
    assert acts[0] == [1, 1]
