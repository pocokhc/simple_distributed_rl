import numpy as np
import pytest

from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.np_array import NpArraySpace
from srl.base.spaces.text import TextSpace


def test_space_basic():
    space = NpArraySpace(3, -1.8, 3.1)

    assert space.size == 3
    assert space.shape == (3,)
    np.testing.assert_array_equal(space.low, np.array([-1.8, -1.8, -1.8], np.float32))
    np.testing.assert_array_equal(space.high, np.array([3.1, 3.1, 3.1], np.float32))
    assert space.stype == SpaceTypes.CONTINUOUS
    assert space.dtype == np.float32

    # --- check_val
    assert space.check_val(np.array([0.0, 0.0, 0.0]))
    assert not space.check_val(np.array([0.0, -2.0, 0.0]))
    assert not space.check_val(np.array([0.0, 0.0]))
    assert not space.check_val([0.0, 0.0, 0.0])

    # --- to_str
    assert space.to_str(np.array([0.1, 0.5, 1.0])) == "0.1,0.5,1"

    # --- copy
    c = space.copy()
    assert space == c

    # --- stack
    o = space.create_stack_space(3)
    assert isinstance(o, NpArraySpace)
    assert o == NpArraySpace(3 * 3, -1.8, 3.1)
    v = space.encode_stack([np.array([1, 1, 0]), np.array([1, 1, 0]), np.array([1, 1, 0])])
    assert (v == np.array([1, 1, 0, 1, 1, 0, 1, 1, 0], dtype=np.float32)).all()


def test_space_get_default():
    space = NpArraySpace(3, -1.8, 3.1)
    assert (space.get_default() == np.array([0, 0, 0], dtype=np.float32)).all()
    space = NpArraySpace(3, 1, [2, 5, 3])
    assert (space.get_default() == np.array([1, 1, 1], dtype=np.float32)).all()
    space = NpArraySpace(3, -2, -1)
    assert (space.get_default() == np.array([-2, -2, -2], dtype=np.float32)).all()


def test_sample():
    space = NpArraySpace(3, -1.8, 3.1)
    print(space)

    # --- sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, np.ndarray)
        assert len(action) == 3
        for a in action:
            assert isinstance(a, float)
        assert -1.8 <= action[0] <= 3.1
        assert -1.8 <= action[1] <= 3.1
        assert -1.8 <= action[2] <= 3.1

    # --- eq
    assert space == NpArraySpace(3, -1.8, 3.1)
    assert space != NpArraySpace(3, -1.8, 3.0)


def test_division():
    space = NpArraySpace(3, -1, 3)

    # action discrete
    space.create_division_tbl(5)
    assert space.division_tbl is not None
    print(space.division_tbl)

    assert len(space.division_tbl) == 8
    en = space.encode_to_space_DiscreteSpace(np.array([-1, -1, 2]))
    assert en == 1
    de = space.decode_from_space_DiscreteSpace(2)
    assert (de == np.array([-1.0, 3.0, -1.0], np.float32)).all()


def test_inf():
    space = NpArraySpace(3)

    # sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, np.ndarray)
        assert len(action) == 3
        for a in action:
            assert isinstance(a, float)


def test_rescale_from():
    space = NpArraySpace(3, 0, 1)
    x = np.array([10, 11, 12])
    y = space.rescale_from(x, 10, 12)
    assert (y == np.array([0, 0.5, 1], dtype=np.float32)).all()


def test_sanitize():
    space = NpArraySpace(3, -1, 3)

    val = space.sanitize(1)
    assert space.check_val(val)
    np.testing.assert_array_equal(np.array([1.0, 1.0, 1.0], np.float32), val)

    val = space.sanitize([1.2, 0.9, 0.8])
    assert space.check_val(val)
    np.testing.assert_array_equal(np.array([1.2, 0.9, 0.8], np.float32), val)

    val = space.sanitize((2, 1, True))
    assert space.check_val(val)
    np.testing.assert_array_equal(np.array([2.0, 1.0, 1.0], np.float32), val)

    val = space.sanitize(np.array([1.2, 0.9, 0.8]))
    assert space.check_val(val)
    np.testing.assert_array_equal(np.array([1.2, 0.9, 0.8], np.float32), val)

    assert not space.check_val([1])
    assert not space.check_val(np.array([1], np.float32))
    assert not space.check_val(np.array([-2.1, 1.1, 1.1], np.float32))
    assert not space.check_val(np.array([5.1, 1.1, 1.1], np.float32))


def test_sanitize2():
    space = NpArraySpace(3, -1, 3)

    val = space.sanitize([-2, 6, 2])
    assert space.check_val(val)
    np.testing.assert_array_equal([-1, 3, 2], val)


@pytest.mark.parametrize(
    "create_space, true_space, val, decode_val",
    [
        [RLBaseTypes.NONE, NpArraySpace(2, -1, 3), np.array([1.1, 1.0], np.float32), [1.1, 1.0]],
        [RLBaseTypes.DISCRETE, DiscreteSpace(4, 0), 2, [3.0, -1.0]],
        [RLBaseTypes.ARRAY_DISCRETE, ArrayDiscreteSpace(2, -1, 3), [1, 1], [1.0, 1.0]],
        [RLBaseTypes.CONTINUOUS, None, 1.0, [1.0, 1.0]],
        [RLBaseTypes.ARRAY_CONTINUOUS, ArrayContinuousSpace(2, -1, 3), [1.1, 1.0], [1.1, 1.0]],
        [RLBaseTypes.NP_ARRAY, NpArraySpace(2, -1, 3), np.array([1.1, 1.0], np.float32), [1.1, 1.0]],
        [RLBaseTypes.NP_ARRAY_UNTYPED, NpArraySpace(2, -1, 3), np.array([1.1, 1.0], np.float32), [1.1, 1.0]],
        [RLBaseTypes.BOX, BoxSpace((2,), -1, 3), np.full((2,), 2.0), [2.0, 2.0]],
        [RLBaseTypes.BOX_UNTYPED, BoxSpace((2,), -1, 3), np.full((2,), 2.0), [2.0, 2.0]],
        [RLBaseTypes.TEXT, TextSpace(min_length=1, charset="0123456789-.,"), "2.0,1.0", [2.0, 1.0]],
    ],
)
def test_space(create_space, true_space, val, decode_val):
    decode_val = np.array(decode_val, dtype=np.float32)
    space = NpArraySpace(2, -1, 3)
    print(space)

    if true_space is None:
        with pytest.raises(NotSupportedError):
            space.create_encode_space(create_space)
        return

    if create_space in [RLBaseTypes.DISCRETE]:
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
    elif create_space == RLBaseTypes.ARRAY_CONTINUOUS:
        assert np.allclose(en, val)
    else:
        assert en == val
    assert target_space.check_val(en)

    de = space.decode_from_space(en, target_space)
    if isinstance(de, np.ndarray):
        assert (de == decode_val).all()
    else:
        assert de == decode_val
    assert space.check_val(de)
