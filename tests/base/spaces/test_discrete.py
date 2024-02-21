import numpy as np

from srl.base.define import RLTypes
from srl.base.spaces.discrete import DiscreteSpace


def test_space():
    space = DiscreteSpace(5, start=1)
    assert space.rl_type == RLTypes.DISCRETE
    print(space)

    # --- discrete
    assert space.n == 5
    de = space.decode_from_int(2)
    assert isinstance(de, int)
    assert de == 3
    en = space.encode_to_int(3)
    assert isinstance(en, int)
    assert en == 2

    # --- discrete numpy
    de = space.decode_from_list_int([2])
    assert isinstance(de, int)
    assert de == 3
    en = space.encode_to_list_int(3)
    assert isinstance(en, list)
    assert len(en) == 1
    assert en[0] == 2

    # --- continuous list
    assert space.list_size == 1
    np.testing.assert_array_equal(space.list_low, [0])
    np.testing.assert_array_equal(space.list_high, [4])
    de = space.decode_from_list_float([3.3, 1.2])
    assert isinstance(de, int)
    assert de == 4
    en = space.encode_to_list_float(3)
    assert isinstance(en, list)
    for n in en:
        assert isinstance(n, float)
    assert en[0] == 2.0

    # --- continuous numpy
    assert space.shape == (1,)
    np.testing.assert_array_equal(space.low, [0])
    np.testing.assert_array_equal(space.high, [4])
    de = space.decode_from_np(np.array([3.3, 1.2]))
    assert isinstance(de, int)
    assert de == 4
    en = space.encode_to_np(3, np.float32)
    np.testing.assert_array_equal(en, [2.0])

    # --- sample
    actions = [space.sample([3]) for _ in range(100)]
    actions = sorted(list(set(actions)))
    np.testing.assert_array_equal(actions, [1, 2, 4, 5])

    # --- eq
    assert space == DiscreteSpace(5, start=1)
    assert space != DiscreteSpace(5, start=0)


def test_sanitize():
    space = DiscreteSpace(5, start=1)

    val = space.sanitize([0.9])
    assert space.check_val(val)
    assert val == 1

    val = space.sanitize(0.9)
    assert space.check_val(val)
    assert val == 1

    val = space.sanitize(4)
    assert space.check_val(val)
    assert val == 4

    val = space.sanitize([0.9])
    assert space.check_val(val)
    assert val == 1

    val = space.sanitize((0.9,))
    assert space.check_val(val)
    assert val == 1

    assert not space.check_val(1.1)
    assert not space.check_val(-1)
    assert not space.check_val(0)
    assert not space.check_val(6)


def test_sanitize2():
    space = DiscreteSpace(5, start=1)

    val = space.sanitize([-1])
    assert space.check_val(val)
    assert val == 1

    val = space.sanitize((6, 99))
    assert space.check_val(val)
    assert val == 5


def test_sample():
    space = DiscreteSpace(2, start=1)
    # 1しか出ない
    r = [space.sample([2]) for _ in range(100)]
    assert sum(r) == 100


def test_valid_actions():
    space = DiscreteSpace(5, start=1)
    acts = space.get_valid_actions([2, 3])
    assert acts == [1, 4, 5]
