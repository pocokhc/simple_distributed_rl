import numpy as np
import pytest

from srl.base import spaces
from srl.base.define import SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.continuous import ContinuousSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.multi import MultiSpace


def test_space_basic():
    space = MultiSpace(
        [
            DiscreteSpace(5),
            ArrayDiscreteSpace(2, 0, 5),
            ContinuousSpace(0, 2),
            ArrayContinuousSpace(2, 0, 2),
            BoxSpace((2, 1), 0, 2),
        ]
    )

    assert space.space_size == 5
    assert space.stype == SpaceTypes.MULTI
    with pytest.raises(NotSupportedError):
        print(space.dtype)

    # --- check_val
    assert space.check_val([1, [1, 1], 1.2, [1.1, 1.2], np.array([[1.1], [1.2]])])
    assert not space.check_val([1, [1, 1], 1.2, [1.1, 1.2], np.array([[1.1], [3.2]])])

    # --- to_str
    _s = "1_1,1_1.2_1.1,1.2_1.1,1.2"
    assert space.to_str([1, [1, 1], 1.2, [1.1, 1.2], np.array([[1.1], [1.2]])]) == _s

    # --- copy
    c = space.copy()
    assert space == c

    # --- get_default
    v = space.get_default()
    assert len(v) == 5
    assert v[0] == 0
    assert v[1] == [0, 0]
    assert v[2] == 0.0
    assert v[3] == [0.0, 0.0]
    assert (v[4] == np.array([[0], [0]])).all()

    # --- stack
    o = space.create_stack_space(3)
    stack_space = MultiSpace(
        [
            ArrayDiscreteSpace(3, 0, 4),
            ArrayDiscreteSpace(2 * 3, 0, 5),
            ArrayContinuousSpace(3, 0, 2),
            ArrayContinuousSpace(2 * 3, 0, 2),
            BoxSpace((3, 2, 1), 0, 2),
        ]
    )
    assert o == stack_space
    v = space.encode_stack([space.sample(), space.sample(), space.sample()])
    assert stack_space.check_val(v)


def test_space_encode():
    space = MultiSpace(
        [
            DiscreteSpace(2),
            ArrayDiscreteSpace(2, 0, 1),
            ContinuousSpace(0, 1),
            ArrayContinuousSpace(2, 0, 1),
            BoxSpace((2, 1), 0, 1),
        ]
    )
    print(space)
    assert space.stype == SpaceTypes.MULTI

    space.create_division_tbl(2)

    # --- discrete
    assert space.int_size == 2 * (2 * 2) * 2 * (2 * 2) * (2 * 2)
    de = space.decode_from_int(0)
    assert len(de) == 5
    assert de[0] == 0
    assert de[1] == [0, 0]
    assert de[2] == 0
    assert de[3] == [0, 0]
    assert (de[4] == np.array([[0], [0]])).all()
    en = space.encode_to_int(de)
    assert isinstance(en, int)
    assert en == 0

    # --- list int
    assert space.list_int_size == 1 + 2 + 1 + 1 + 1
    assert space.list_int_low == [0] + [0, 0] + [0] + [0] + [0]
    assert space.list_int_high == [1] + [1, 1] + [2] + [4] + [4]
    x = [0] + [0, 0] + [0] + [0] + [0]
    de = space.decode_from_list_int(x)
    assert len(de) == 5
    assert de[0] == 0
    assert de[1] == [0, 0]
    assert de[2] == 0
    assert de[3] == [0, 0]
    assert (de[4] == np.array([[0], [0]])).all()
    en = space.encode_to_list_int(de)
    assert len(en) == 1 + 2 + 1 + 1 + 1
    np.testing.assert_array_equal(en, x)

    # --- list float
    assert space.list_float_size == 1 + 2 + 1 + 2 + 2
    assert space.list_float_low == [0] + [0, 0] + [0] + [0, 0] + [0, 0]
    assert space.list_float_high == [1] + [1, 1] + [1] + [1, 1] + [1, 1]
    de = space.decode_from_list_float([0.0] + [0, 0] + [0] + [0, 0] + [0, 0])
    assert len(de) == 5
    assert de[0] == 0
    assert de[1] == [0, 0]
    assert de[2] == 0
    assert de[3] == [0, 0]
    assert (de[4] == np.array([[0], [0]])).all()
    en = space.encode_to_list_float(de)
    assert len(en) == 1 + 2 + 1 + 2 + 2
    np.testing.assert_array_equal(en, [0] + [0, 0] + [0] + [0, 0] + [0, 0])

    # --- np
    assert space.np_shape == (1 + 2 + 1 + 2 + 2,)
    assert (space.np_low == np.array([0] + [0, 0] + [0] + [0, 0] + [0, 0])).all()
    assert (space.np_high == np.array([1] + [1, 1] + [1] + [1, 1] + [1, 1])).all()
    de = space.decode_from_np(np.array([0.0] + [0, 0] + [0] + [0, 0] + [0, 0]))
    assert len(de) == 5
    assert de[0] == 0
    assert de[1] == [0, 0]
    assert de[2] == 0
    assert de[3] == [0, 0]
    assert (de[4] == np.array([[0], [0]])).all()
    en = space.encode_to_np(de, np.float32)
    assert (en == np.array([0.0] + [0, 0] + [0] + [0, 0] + [0, 0], np.float32)).all()


def test_sanitize():
    space = MultiSpace(
        [
            DiscreteSpace(2),
            ArrayDiscreteSpace(2, 0, 1),
            ContinuousSpace(0, 1),
            ArrayContinuousSpace(2, 0, 1),
            BoxSpace((2, 1), 0, 1),
        ]
    )
    val = space.sanitize(
        [
            [3],
            [3, 3],
            [3],
            [3, 3],
            [3, 3],
        ]
    )
    print(val)
    assert len(val) == 5
    assert space.spaces[0].check_val(val[0])
    assert space.spaces[1].check_val(val[1])
    assert space.spaces[2].check_val(val[2])
    assert space.spaces[3].check_val(val[3])
    assert space.spaces[4].check_val(val[4])


def test_sample():
    space = MultiSpace(
        [
            DiscreteSpace(5),
            ArrayDiscreteSpace(2, 0, 5),
            ContinuousSpace(0, 2),
            ArrayContinuousSpace(2, 0, 2),
            BoxSpace((2, 1), 0, 2),
        ]
    )
    for _ in range(100):
        act = space.sample()
        assert len(act) == 5
        assert space.spaces[0].check_val(act[0])
        assert space.spaces[1].check_val(act[1])
        assert space.spaces[2].check_val(act[2])
        assert space.spaces[3].check_val(act[3])
        assert space.spaces[4].check_val(act[4])


def test_valid_actions():
    space = MultiSpace(
        [
            DiscreteSpace(5),
            ArrayDiscreteSpace(2, 0, 5),
            ContinuousSpace(0, 2),
            ArrayContinuousSpace(2, 0, 2),
            BoxSpace((2, 1), 0, 2),
        ]
    )
    with pytest.raises(NotSupportedError):
        space.get_valid_actions()

    space = MultiSpace(
        [
            DiscreteSpace(3),
            ArrayDiscreteSpace(2, 0, 1),
            BoxSpace((1, 1), 0, 1, np.int8),
        ]
    )
    acts = space.get_valid_actions()
    assert len(acts) == 3 * (2 * 2) * 2

    space = MultiSpace(
        [
            DiscreteSpace(2),
            ArrayDiscreteSpace(1, 0, 1),
            BoxSpace((1, 1), 0, 1, np.int8),
        ]
    )
    acts = space.get_valid_actions(
        [
            [0, [0], np.array([[0]], dtype=np.int8)],
            [0, [1], np.array([[0]], dtype=np.int8)],
            [0, [0], np.array([[1]], dtype=np.int8)],
            [0, [1], np.array([[1]], dtype=np.int8)],
        ]
    )
    print(acts)
    assert len(acts) == 4
    assert len(acts[0]) == 3
    assert acts[0][0] == 1
    assert acts[0][1] == [0]
    assert (acts[0][2] == [[0]]).all()


@pytest.mark.parametrize(
    "create_space, true_space, val, decode_val",
    [
        ["", spaces.DiscreteSpace(5, 0), 2, 3],
        ["DiscreteSpace", spaces.DiscreteSpace(5, 0), 2, 3],
        ["ArrayDiscreteSpace", spaces.ArrayDiscreteSpace(1, 0, 4), [2], 3],
        ["ContinuousSpace", spaces.ContinuousSpace(0, 4), 2, 3],
        ["ArrayContinuousSpace", spaces.ArrayContinuousSpace(1, 0, 4), [2], 3],
        ["BoxSpace", spaces.BoxSpace((1,), 0, 4, np.int64), np.full((1,), 2), 3],
        ["BoxSpace_float", spaces.BoxSpace((1,), 0, 4, np.float32), np.full((1,), 2), 3],
        ["TextSpace", None, "2", 3],
    ],
)
def test_space(create_space, true_space, val, decode_val):
    pytest.skip("TODO")
