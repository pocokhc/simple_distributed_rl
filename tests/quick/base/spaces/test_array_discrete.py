import numpy as np
import pytest

from srl.base import spaces
from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.np_array import NpArraySpace
from srl.base.spaces.space import SpaceEncodeOptions
from srl.base.spaces.text import TextSpace


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

    # --- stack
    o = space.create_stack_space(3)
    assert isinstance(o, ArrayDiscreteSpace)
    assert o == ArrayDiscreteSpace(3 * 3, 0, [2, 5, 3] * 3)
    v = space.encode_stack([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    assert v == [1, 1, 0, 1, 1, 0, 1, 1, 0]


def test_space_get_default():
    space = ArrayDiscreteSpace(3, 0, [2, 5, 3])

    assert space.get_default() == [0, 0, 0]
    space = ArrayDiscreteSpace(3, 1, [2, 5, 3])
    assert space.get_default() == [1, 1, 1]
    space = ArrayDiscreteSpace(3, -2, -1)
    assert space.get_default() == [-2, -2, -2]


def test_get_onehot():
    space = ArrayDiscreteSpace(3, 0, [2, 5, 3])

    # Valid input
    onehot = space.get_onehot([1, 2, 0])
    assert isinstance(onehot, list)
    assert all(isinstance(inner, list) for inner in onehot)
    assert all(isinstance(value, int) for inner in onehot for value in inner)

    # Check onehot encoding correctness
    assert onehot == [
        [0, 1, 0],  # One-hot for 1 in range [0, 2]
        [0, 0, 1, 0, 0, 0],  # One-hot for 2 in range [0, 5]
        [1, 0, 0, 0],  # One-hot for 0 in range [0, 3]
    ]

    # Edge cases
    onehot = space.get_onehot([0, 5, 3])
    assert onehot == [
        [1, 0, 0],  # One-hot for 0 in range [0, 2]
        [0, 0, 0, 0, 0, 1],  # One-hot for 5 in range [0, 5]
        [0, 0, 0, 1],  # One-hot for 3 in range [0, 3]
    ]

    # Invalid input (out of bounds)
    with pytest.raises(ValueError):
        space.get_onehot([3, 2, 1])  # 3 is out of bounds for the first dimension


def test_sample():
    space = ArrayDiscreteSpace(3, 0, [2, 5, 3])
    print(space)

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


def test_sample2():
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


@pytest.mark.parametrize(
    "create_space, options, true_space, val, decode_val",
    [
        [RLBaseTypes.NONE, None, spaces.ArrayDiscreteSpace(2, -1, 3), [1, 1], [1, 1]],
        [RLBaseTypes.DISCRETE, None, spaces.DiscreteSpace(25), 2, [-1, 1]],
        [RLBaseTypes.ARRAY_DISCRETE, None, spaces.ArrayDiscreteSpace(2, -1, 3), [1, 1], [1, 1]],
        [RLBaseTypes.CONTINUOUS, None, None, 1.0, [1, 1]],
        [RLBaseTypes.ARRAY_CONTINUOUS, None, spaces.ArrayContinuousSpace(2, -1, 3), [1.0, 1.0], [1, 1]],
        [RLBaseTypes.NP_ARRAY, None, NpArraySpace(2, -1, 3, np.float32, SpaceTypes.DISCRETE), np.array([1.0, 1.0], np.float32), [1, 1]],
        [RLBaseTypes.NP_ARRAY, SpaceEncodeOptions(cast=False), NpArraySpace(2, -1, 3, np.int64), np.array([1, 1], np.int64), [1, 1]],
        [RLBaseTypes.BOX, None, spaces.BoxSpace((2,), -1, 3, np.float32, SpaceTypes.DISCRETE), np.full((2,), 2.0, np.float32), [2, 2]],
        [RLBaseTypes.BOX, SpaceEncodeOptions(cast=False), spaces.BoxSpace((2,), -1, 3, np.int64), np.full((2,), 2), [2, 2]],
        [RLBaseTypes.TEXT, None, TextSpace(min_length=1, charset="0123456789,"), "2,1", [2, 1]],
    ],
)
def test_space(create_space, options, true_space, val, decode_val):
    space = ArrayDiscreteSpace(2, -1, 3)
    print(space)
    if options is None:
        options = SpaceEncodeOptions(cast=True)

    if true_space is None:
        with pytest.raises(NotSupportedError):
            space.create_encode_space(create_space, options)
        return

    target_space = space.create_encode_space(create_space, options)
    print(target_space)
    print(true_space)
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
