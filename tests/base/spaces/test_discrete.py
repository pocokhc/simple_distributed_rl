import numpy as np

from srl.base.define import RLTypes
from srl.base.spaces.discrete import DiscreteSpace


def test_space():
    space = DiscreteSpace(5)
    assert space.rl_type == RLTypes.DISCRETE
    print(space)

    # --- discrete
    assert space.n == 5
    de = space.decode_from_int(2)
    assert isinstance(de, int)
    assert de == 2
    en = space.encode_to_int(3)
    assert isinstance(en, int)
    assert en == 3

    # --- discrete numpy
    de = space.decode_from_int_np(np.array([2]))
    assert isinstance(de, int)
    assert de == 2
    en = space.encode_to_int_np(3)
    assert isinstance(en, np.ndarray)
    assert en.shape == (1,)
    assert en[0] == 3

    # --- continuous list
    assert space.list_size == 1
    np.testing.assert_array_equal(space.list_low, [0])
    np.testing.assert_array_equal(space.list_high, [4])
    de = space.decode_from_list_float([3.3, 1.2])
    assert isinstance(de, int)
    assert de == 3
    en = space.encode_to_list_float(3)
    assert isinstance(en, list)
    for n in en:
        assert isinstance(n, float)
    assert en[0] == 3.0

    # --- continuous numpy
    assert space.shape == (1,)
    np.testing.assert_array_equal(space.low, [0])
    np.testing.assert_array_equal(space.high, [4])
    de = space.decode_from_np(np.array([3.3, 1.2]))
    assert isinstance(de, int)
    assert de == 3
    en = space.encode_to_np(3)
    np.testing.assert_array_equal(en, [3.0])

    # --- sample
    actions = [space.sample([3]) for _ in range(100)]
    actions = sorted(list(set(actions)))
    np.testing.assert_array_equal(actions, [0, 1, 2, 4])

    # --- eq
    assert space == DiscreteSpace(5)
    assert space != DiscreteSpace(4)


def test_convert():
    space = DiscreteSpace(5)

    val = space.convert([0.9])
    assert space.check_val(val)
    assert val == 1

    val = space.convert(0.9)
    assert space.check_val(val)
    assert val == 1

    val = space.convert(4)
    assert space.check_val(val)
    assert val == 4

    val = space.convert([0.9])
    assert space.check_val(val)
    assert val == 1

    val = space.convert((0.9,))
    assert space.check_val(val)
    assert val == 1

    assert not space.check_val(1.1)
    assert not space.check_val(-1)
    assert not space.check_val(5)


def test_convert2():
    space = DiscreteSpace(5)

    val = space.convert([-1])
    assert space.check_val(val)
    assert val == 0

    val = space.convert((6, 99))
    assert space.check_val(val)
    assert val == 4
