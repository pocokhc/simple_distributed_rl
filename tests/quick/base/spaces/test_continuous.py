import numpy as np
import pytest

from srl.base import spaces
from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.continuous import ContinuousSpace
from srl.base.spaces.np_array import NpArraySpace
from srl.base.spaces.space import SpaceEncodeOptions
from srl.base.spaces.text import TextSpace


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


def test_sample():
    space = ContinuousSpace(-1.8, 3.1)
    print(space)

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


def test_inf():
    space = ContinuousSpace()

    print(space)

    # sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, float)


def test_rescale_from():
    space = ContinuousSpace(0, 1)
    x = 10.0
    y = space.rescale_from(x, 10, 12)
    assert y == 0.0


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
    "create_space, options, true_space, val, decode_val",
    [
        [RLBaseTypes.NONE, None, ContinuousSpace(-1, 3), 1.1, 1.1],
        [RLBaseTypes.DISCRETE, None, spaces.DiscreteSpace(5, 0), 1, 0],
        [RLBaseTypes.ARRAY_DISCRETE, None, spaces.ArrayDiscreteSpace(1, -1, 3), [2], 2.0],
        [RLBaseTypes.CONTINUOUS, None, spaces.ContinuousSpace(-1, 3), 1.1, 1.1],
        [RLBaseTypes.ARRAY_CONTINUOUS, None, spaces.ArrayContinuousSpace(1, -1, 3), [1.1], 1.1],
        [RLBaseTypes.NP_ARRAY, None, NpArraySpace(1, -1, 3), np.array([1.25], np.float32), 1.25],
        [RLBaseTypes.NP_ARRAY, SpaceEncodeOptions(cast=False), NpArraySpace(1, -1, 3), np.array([1.25], np.float32), 1.25],
        [RLBaseTypes.BOX, None, spaces.BoxSpace((1,), -1, 3), np.full((1,), 0.25, np.float32), 0.25],
        [RLBaseTypes.BOX, SpaceEncodeOptions(cast=False), spaces.BoxSpace((1,), -1, 3), np.full((1,), 0.25, np.float32), 0.25],
        [RLBaseTypes.TEXT, None, TextSpace(min_length=1, charset="0123456789-."), "-0.5", -0.5],
    ],
)
def test_space(create_space, options, true_space, val, decode_val):
    space = ContinuousSpace(-1, 3)
    print(space)
    if options is None:
        options = SpaceEncodeOptions(cast=True)

    if true_space is None:
        with pytest.raises(NotSupportedError):
            space.set_encode_space(create_space, options)
        return

    if create_space in [RLBaseTypes.DISCRETE]:
        space.create_division_tbl(5)
    target_space = space.set_encode_space(create_space, options)
    print(target_space)
    assert target_space == true_space

    de = space.decode_from_space(val)
    print(de)
    if isinstance(de, np.ndarray):
        assert (de == decode_val).all()
    else:
        assert de == decode_val
    assert space.check_val(de)
    en = space.encode_to_space(decode_val)
    if isinstance(en, np.ndarray):
        assert (en == val).all()
    else:
        assert en == val
    assert target_space.check_val(en)

    de = space.decode_from_space(en)
    if isinstance(de, np.ndarray):
        assert (de == decode_val).all()
    else:
        assert de == decode_val
    assert space.check_val(de)


@pytest.mark.parametrize(
    "create_space, options, true_space, val, decode_val",
    [
        [RLBaseTypes.NP_ARRAY, SpaceEncodeOptions(np_zero_start=True), NpArraySpace(1, 0, 4, np.float32), np.array([4.0]), 3],
        [RLBaseTypes.NP_ARRAY, SpaceEncodeOptions(np_norm_type="0to1"), NpArraySpace(1, 0, 1, np.float32), np.array([1.0]), 3],
        [RLBaseTypes.NP_ARRAY, SpaceEncodeOptions(np_norm_type="-1to1"), NpArraySpace(1, -1, 1, np.float32), np.array([-1.0]), -1],
        [RLBaseTypes.BOX, SpaceEncodeOptions(np_zero_start=True), BoxSpace((1,), 0, 4, np.float32), np.full((1,), 0.0), -1],
        [RLBaseTypes.BOX, SpaceEncodeOptions(np_norm_type="0to1"), BoxSpace((1,), 0, 1, np.float32), np.array([1.0]), 3],
        [RLBaseTypes.BOX, SpaceEncodeOptions(np_norm_type="-1to1"), BoxSpace((1,), -1, 1, np.float32), np.array([-1.0]), -1],
    ],
)
def test_space_zero_start(create_space, options, true_space, val, decode_val):
    space = ContinuousSpace(-1, 3)
    print(space)

    target_space = space.set_encode_space(create_space, options)
    print(target_space)
    print(true_space)
    assert target_space == true_space

    de = space.decode_from_space(val)
    print(de)
    if isinstance(de, np.ndarray):
        assert (de == decode_val).all()
    else:
        assert de == decode_val
    assert space.check_val(de)
    en = space.encode_to_space(decode_val)
    if isinstance(en, np.ndarray):
        assert (en == val).all()
    else:
        assert en == val
    assert target_space.check_val(en)

    de = space.decode_from_space(en)
    if isinstance(de, np.ndarray):
        assert (de == decode_val).all()
    else:
        assert de == decode_val
    assert space.check_val(de)
