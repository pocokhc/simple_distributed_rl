import numpy as np

from srl.base.define import RLTypes
from srl.base.spaces import ArrayDiscreteSpace


def test_space():
    space = ArrayDiscreteSpace(3, 0, [2, 5, 3])
    assert space.rl_type == RLTypes.DISCRETE

    print(space)

    # --- discrete
    assert space.n == (2 + 1) * (5 + 1) * (3 + 1)
    de = space.decode_from_int(1)
    assert isinstance(de, list)
    for n in de:
        assert isinstance(n, int)
    np.testing.assert_array_equal(de, [0, 0, 1])
    en = space.encode_to_int([0, 0, 1])
    assert isinstance(en, int)
    assert en == 1

    # --- discrete numpy
    de = space.decode_from_list_int([1, 2, 0])
    assert isinstance(de, list)
    for n in de:
        assert isinstance(n, int)
    np.testing.assert_array_equal(de, [1, 2, 0])
    en = space.encode_to_list_int([1, 2, 0])
    assert isinstance(en, list)
    assert len(en) == 3
    np.testing.assert_array_equal(en, [1, 2, 0])

    # --- continuous list
    assert space.list_size == 3
    np.testing.assert_array_equal(space.list_low, [0, 0, 0])
    np.testing.assert_array_equal(space.list_high, [2, 5, 3])
    de = space.decode_from_list_float([0.1, 0.6, 0.9])
    assert isinstance(de, list)
    for n in de:
        assert isinstance(n, int)
    np.testing.assert_array_equal(de, [0, 1, 1])
    en = space.encode_to_list_float([0, 1, 1])
    assert isinstance(en, list)
    for n in en:
        assert isinstance(n, float)
    np.testing.assert_array_equal(en, [0.0, 1.0, 1.0])

    # --- continuous numpy
    assert space.shape == (3,)
    np.testing.assert_array_equal(space.low, [0, 0, 0])
    np.testing.assert_array_equal(space.high, [2, 5, 3])
    de = space.decode_from_np(np.array([0.1, 0.6, 0.9]))
    assert isinstance(de, list)
    for n in de:
        assert isinstance(n, int)
    np.testing.assert_array_equal(de, [0, 1, 1])
    en = space.encode_to_np([0, 1, 1], dtype=np.float32)
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
