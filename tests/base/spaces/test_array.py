import numpy as np

from srl.base.define import RLTypes
from srl.base.spaces import ArrayDiscreteSpace
from srl.base.spaces.array import ArraySpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.discrete import DiscreteSpace


def test_space_discrete():
    space = ArraySpace(
        [
            DiscreteSpace(5),
            ArrayDiscreteSpace(2, 0, [1, 2]),
            BoxSpace((2, 1), -1, 0, dtype=np.int64),
        ]
    )
    print(space)
    assert space.rl_type == RLTypes.DISCRETE

    # --- discrete
    assert space.n == 5 * ((1 + 1) * (2 + 1)) * (2 * 2)
    de = space.decode_from_int(1)
    assert len(de) == 3
    assert isinstance(de[0], int)
    assert de[0] == 0
    assert isinstance(de[1], list)
    assert len(de[1]) == 2
    assert de[1] == [0, 0]
    assert isinstance(de[2], np.ndarray)
    assert de[2].shape == (2, 1)
    np.testing.assert_array_equal(de[2], np.array([[-1], [0]], dtype=np.int64))
    en = space.encode_to_int([0, [0, 0], np.array([[-1], [0]], dtype=np.int64)])
    assert isinstance(en, int)
    assert en == 1

    # --- discrete list
    en = space.encode_to_list_int([0, [0, 0], np.array([[-1], [0]], dtype=np.int64)])
    assert len(en) == 5
    np.testing.assert_array_equal(en, [0, 0, 0, -1, 0])
    de = space.decode_from_list_int([0, 0, 0, -1, 0])
    assert len(de) == 3
    assert isinstance(de[0], int)
    assert de[0] == 0
    assert isinstance(de[1], list)
    assert len(de[1]) == 2
    assert de[1] == [0, 0]
    assert isinstance(de[2], np.ndarray)
    assert de[2].shape == (2, 1)
    np.testing.assert_array_equal(de[2], np.array([[-1], [0]], dtype=np.int64))

    # --- continuous list
    en = space.encode_to_list_float([0, [0, 0], np.array([[-1], [0]], dtype=np.int64)])
    assert isinstance(en, list)
    for n in en:
        assert isinstance(n, float)
    np.testing.assert_array_equal(en, [0, 0, 0, -1, 0])
    assert space.list_size == 5
    np.testing.assert_array_equal(space.list_low, [0, 0, 0, -1, -1])
    np.testing.assert_array_equal(space.list_high, [4, 1, 2, 0, 0])
    de = space.decode_from_list_float([0, 0, 0, -1, 0])
    assert len(de) == 3
    assert isinstance(de[0], int)
    assert de[0] == 0
    assert isinstance(de[1], list)
    assert len(de[1]) == 2
    assert de[1] == [0, 0]
    assert isinstance(de[2], np.ndarray)
    assert de[2].shape == (2, 1)
    np.testing.assert_array_equal(de[2], np.array([[-1], [0]], dtype=np.int64))

    # --- continuous numpy
    en = space.encode_to_np([0, [0, 0], np.array([[-1], [0]], dtype=np.int64)], np.float32)
    assert len(en) == 3
    np.testing.assert_array_equal(en[0], np.array([0], np.float32))
    np.testing.assert_array_equal(en[1], np.array([0, 0], np.float32))
    np.testing.assert_array_equal(en[2], np.array([[-1], [0]], np.float32))
    assert space.shape[0] == (1,)
    assert space.shape[1] == (2,)
    assert space.shape[2] == (2, 1)
    np.testing.assert_array_equal(space.low[0], [0])
    np.testing.assert_array_equal(space.low[1], [0, 0])
    np.testing.assert_array_equal(space.low[2], [[-1], [-1]])
    np.testing.assert_array_equal(space.high[0], [4])
    np.testing.assert_array_equal(space.high[1], [1, 2])
    np.testing.assert_array_equal(space.high[2], [[0], [0]])
    de = space.decode_from_np(en)
    assert len(de) == 3
    assert isinstance(de[0], int)
    assert de[0] == 0
    assert isinstance(de[1], list)
    assert len(de[1]) == 2
    assert de[1] == [0, 0]
    assert isinstance(de[2], np.ndarray)
    assert de[2].shape == (2, 1)
    np.testing.assert_array_equal(de[2], np.array([[-1], [0]], dtype=np.int64))

    # --- sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, list)
        assert len(action) == 3

    # --- eq
    assert space == ArraySpace(
        [
            DiscreteSpace(5),
            ArrayDiscreteSpace(2, 0, [1, 2]),
            BoxSpace((2, 1), -1, 0, dtype=np.int64),
        ]
    )
    assert space != ArraySpace(
        [
            DiscreteSpace(5),
            ArrayDiscreteSpace(2, 1, [1, 2]),
            BoxSpace((2, 1), -1, 0, dtype=np.int64),
        ]
    )


def test_sanitize():
    space = ArraySpace(
        [
            DiscreteSpace(5),
            ArrayDiscreteSpace(2, 0, [1, 2]),
            BoxSpace((2, 1), -1, 0, dtype=np.int64),
        ]
    )

    val = space.sanitize([1.2, [1.1, 1.2], [[1], [1]]])
    print(val)
    assert len(val) == 3
    assert space.check_val(val)
    assert val[0] == 1
    assert val[1] == [1, 1]
    np.testing.assert_array_equal(val[2], [[0], [0]])


def test_sample_discrete():
    space = ArraySpace(
        [
            DiscreteSpace(2),
            ArrayDiscreteSpace(1, 0, 1),
            BoxSpace((1, 1), -1, 0, dtype=np.int64),
        ]
    )
    for _ in range(100):
        r = space.sample(
            [
                [0, [0], np.array([[0]], dtype=np.int64)],
                [0, [1], np.array([[0]], dtype=np.int64)],
                [0, [0], np.array([[-1]], dtype=np.int64)],
                [0, [1], np.array([[-1]], dtype=np.int64)],
                [1, [0], np.array([[0]], dtype=np.int64)],
                [1, [1], np.array([[0]], dtype=np.int64)],
                [1, [0], np.array([[-1]], dtype=np.int64)],
            ]
        )
        assert len(r) == 3
        assert r[0] == 1
        assert r[1] == [1]
        assert (r[2] == [[-1]]).all()


def test_valid_actions():
    space = ArraySpace(
        [
            DiscreteSpace(2),
            ArrayDiscreteSpace(1, 0, 1),
            BoxSpace((1, 1), -1, 0, dtype=np.int64),
        ]
    )
    acts = space.get_valid_actions(
        [
            [0, [0], np.array([[0]], dtype=np.int64)],
            [0, [1], np.array([[0]], dtype=np.int64)],
            [0, [0], np.array([[-1]], dtype=np.int64)],
            [0, [1], np.array([[-1]], dtype=np.int64)],
            [1, [0], np.array([[0]], dtype=np.int64)],
            [1, [1], np.array([[0]], dtype=np.int64)],
            [1, [0], np.array([[-1]], dtype=np.int64)],
        ]
    )
    print(acts)
    assert len(acts) == 1
    assert len(acts[0]) == 3
    assert acts[0][0] == 1
    assert acts[0][1] == [1]
    assert (acts[0][2] == [[-1]]).all()
