import numpy as np
import pytest

from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.continuous import ContinuousSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.np_array import NpArraySpace
from srl.base.spaces.text import TextSpace


def test_space_basic():
    space = DiscreteSpace(5, start=1)

    assert space.n == 5
    assert space.start == 1
    assert space.stype == SpaceTypes.DISCRETE

    # --- check_val
    assert space.check_val(3)
    assert not space.check_val(0)

    # --- to_str
    assert space.to_str(3) == "3"

    # --- str
    assert str(space) == "Discrete(5, start=1)"

    # --- get_default
    assert space.get_default() == 1

    # --- copy
    c = space.copy()
    assert space == c
    assert space is not c

    # --- stack
    o = space.create_stack_space(3)
    assert isinstance(o, ArrayDiscreteSpace)
    assert o == ArrayDiscreteSpace(3, 1, 5)
    v = space.encode_stack([1, 1, 0])
    assert v == [1, 1, 0]


def test_dtype():
    space = DiscreteSpace(5, start=1)
    assert space.dtype == np.uint64

    space = DiscreteSpace(5, start=-1)
    assert space.dtype == np.int64


@pytest.mark.parametrize(
    "val,expected",
    [
        (1, False),
        (2, True),
        (6, True),
        (7, False),
        ("a", False),
    ],
)
def test_check_val(val, expected):
    space = DiscreteSpace(5, start=2)
    assert space.check_val(val) is expected


def test_get_onehot():
    space = DiscreteSpace(5, start=1)

    # Test valid input
    onehot = space.get_onehot(3)
    assert isinstance(onehot, list)
    assert len(onehot) == 5
    assert onehot == [0, 0, 1, 0, 0]

    # Test boundary values
    onehot = space.get_onehot(1)
    assert onehot == [1, 0, 0, 0, 0]

    onehot = space.get_onehot(4)
    assert onehot == [0, 0, 0, 1, 0]

    # Test invalid input
    with pytest.raises(IndexError):
        space.get_onehot(0)

    with pytest.raises(IndexError):
        space.get_onehot(6)


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


def test_sample2():
    space = DiscreteSpace(5, start=1)
    print(space)

    # --- sample
    actions = [space.sample([3, 4]) for _ in range(100)]
    actions = sorted(list(set(actions)))
    np.testing.assert_array_equal(actions, [1, 2, 5])

    # --- eq
    assert space == DiscreteSpace(5, start=1)
    assert space != DiscreteSpace(5, start=0)


def test_valid_actions():
    space = DiscreteSpace(5, start=1)
    acts = space.get_valid_actions([2, 3])
    assert acts == [1, 4, 5]


@pytest.mark.parametrize(
    "create_space, true_space, val, decode_val",
    [
        [RLBaseTypes.NONE, DiscreteSpace(5, 1), 2, 2],
        [RLBaseTypes.DISCRETE, DiscreteSpace(5, 0), 2, 3],
        [RLBaseTypes.ARRAY_DISCRETE, ArrayDiscreteSpace(1, 0, 4), [2], 3],
        [RLBaseTypes.CONTINUOUS, ContinuousSpace(0, 4), 2, 3],
        [RLBaseTypes.ARRAY_CONTINUOUS, ArrayContinuousSpace(1, 0, 4), [2], 3],
        [RLBaseTypes.NP_ARRAY, NpArraySpace(1, 0, 4, np.float32, SpaceTypes.DISCRETE), np.array([2.0], np.float32), 3],
        [RLBaseTypes.NP_ARRAY_UNTYPED, NpArraySpace(1, 0, 4, np.uint), np.array([2], np.uint), 3],
        [RLBaseTypes.BOX, BoxSpace((1,), 0, 4, np.float32, SpaceTypes.DISCRETE), np.full((1,), 2.0, np.float32), 3],
        [RLBaseTypes.BOX_UNTYPED, BoxSpace((1,), 0, 4, np.uint), np.full((1,), 2, np.int64), 3],
        [RLBaseTypes.TEXT, TextSpace(1, 1, "0123456789-"), "2", 3],
    ],
)
def test_space(create_space, true_space, val, decode_val):
    space = DiscreteSpace(5, start=1)
    print(space)

    if true_space is None:
        with pytest.raises(NotSupportedError):
            space.create_encode_space(create_space)
        return

    target_space = space.create_encode_space(create_space)
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
