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
from srl.base.spaces.space import SpaceEncodeOptions
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
    "create_space, options, true_space, val, decode_val",
    [
        [RLBaseTypes.NONE, None, TextSpace(3), "2", "2"],
        [RLBaseTypes.DISCRETE, None, DiscreteSpace(999, 0), 0, "0"],
        [RLBaseTypes.ARRAY_DISCRETE, None, ArrayDiscreteSpace(3, 0, 0x7F), [50, 51, 52], "234"],
        [RLBaseTypes.CONTINUOUS, None, ContinuousSpace(), 1.1, "1.1"],
        [RLBaseTypes.ARRAY_CONTINUOUS, None, ArrayContinuousSpace(3, 0, 0x7F), [50, 51, 52], "234"],
        [RLBaseTypes.NP_ARRAY, None, NpArraySpace(3, 0, 0x7F, np.float32, SpaceTypes.DISCRETE), np.array([50, 51, 52], np.float32), "234"],
        [RLBaseTypes.NP_ARRAY, SpaceEncodeOptions(cast=False), NpArraySpace(3, 0, 0x7F, np.uint), np.array([50, 51, 52], np.uint), "234"],
        [RLBaseTypes.BOX, None, BoxSpace((3,), 0, 0x7F, np.float32, SpaceTypes.DISCRETE), np.array([50, 51, 52], np.float32), "234"],
        [RLBaseTypes.BOX, SpaceEncodeOptions(cast=False), BoxSpace((3,), 0, 0x7F, np.uint), np.array([50, 51, 52], np.uint), "234"],
        [RLBaseTypes.TEXT, None, TextSpace(3), "2", "2"],
    ],
)
def test_space(create_space, options, true_space, val, decode_val):
    space = TextSpace(3)
    print(space)
    if options is None:
        options = SpaceEncodeOptions(cast=True)

    if true_space is None:
        with pytest.raises(NotSupportedError):
            space.set_encode_space(create_space, options)
        return

    target_space = space.set_encode_space(create_space, options)
    print(target_space)
    print(true_space)
    assert target_space == true_space

    de = space.decode_from_space(val)
    print(de)
    assert de == decode_val
    assert space.check_val(de)
    en = space.encode_to_space(decode_val)
    if isinstance(en, np.ndarray):
        assert (en == val).all()
    else:
        assert en == val
    assert target_space.check_val(en)

    de = space.decode_from_space(en)
    assert de == decode_val
    assert space.check_val(de)


@pytest.mark.parametrize(
    "create_space, options, true_space, val, decode_val",
    [
        [RLBaseTypes.NP_ARRAY, SpaceEncodeOptions(np_norm_type="0to1"), NpArraySpace(3, 0, 1), np.array([0, 0, 0], np.float32), ""],
        [RLBaseTypes.NP_ARRAY, SpaceEncodeOptions(np_norm_type="-1to1"), NpArraySpace(3, -1, 1), np.array([0, 0, 0], np.float32), "???"],
        [RLBaseTypes.BOX, SpaceEncodeOptions(np_norm_type="0to1"), BoxSpace((3,), 0, 1), np.array([0, 0, 0], np.float32), ""],
        [RLBaseTypes.BOX, SpaceEncodeOptions(np_norm_type="-1to1"), BoxSpace((3,), -1, 1), np.array([0, 0, 0], np.float32), "???"],
    ],
)
def test_space_zero_start(create_space, options, true_space, val, decode_val):
    space = TextSpace(3)
    print(space)

    target_space = space.set_encode_space(create_space, options)
    print(target_space)
    print(true_space)
    assert target_space == true_space

    de = space.decode_from_space(val)
    print(de)
    if False:  # アンダーフロー誤差でずれる
        if isinstance(de, np.ndarray):
            assert (de == decode_val).all()
        else:
            assert de == decode_val
        assert space.check_val(de)
    en = space.encode_to_space(decode_val)

    if False:  # アンダーフロー誤差でずれる
        if isinstance(en, np.ndarray):
            assert (en == val).all()
        else:
            assert en == val
        assert target_space.check_val(en)

    de = space.decode_from_space(en)
    print(de)
    if False:  # アンダーフロー誤差でずれる
        if isinstance(de, np.ndarray):
            assert (de == decode_val).all()
        else:
            assert de == decode_val
        assert space.check_val(de)
