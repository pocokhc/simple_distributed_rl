import numpy as np
import pytest

from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.continuous import ContinuousSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.np_array import NpArraySpace
from srl.base.spaces.space import SpaceEncodeOptions
from srl.base.spaces.text import TextSpace


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
    "create_space, options, true_space, val, decode_val",
    [
        [
            RLBaseTypes.NONE,
            None,
            MultiSpace,
            [1, [0], 0.1, [0.5], np.array([0.2], np.float32), np.ones((1, 1), np.float32), "5"],
            [1, [0], 0.1, [0.5], np.array([0.2], np.float32), np.ones((1, 1), np.float32), "5"],
        ],
        [
            RLBaseTypes.DISCRETE,
            None,
            DiscreteSpace(2916, 0),
            0,
            [0, [0], 0.0, [0.0], np.array([0], dtype=np.float32), np.array([[0]], dtype=np.float32), "0"],
        ],
        [
            RLBaseTypes.ARRAY_DISCRETE,
            None,
            ArrayDiscreteSpace(7, 0, [1, 1, 3, 3, 3, 3, 127]),
            [1, 1, 1, 1, 1, 1, 49],
            [1, [1], 0.5, [0.5], np.array([0.5], dtype=np.float32), np.array([[0.5]], dtype=np.float32), "1"],
        ],
        [RLBaseTypes.CONTINUOUS, None, None, 1.1, 1.1],
        [
            RLBaseTypes.ARRAY_CONTINUOUS,
            None,
            ArrayContinuousSpace(7, 0, [1, 1, 1, 1, 1, 1, 127]),
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 49.0],
            [1, [1], 1.0, [1.0], np.array([1.0], dtype=np.float32), np.array([[1.0]], dtype=np.float32), "1"],
        ],
        [
            RLBaseTypes.NP_ARRAY,
            None,
            NpArraySpace(7, 0, [1, 1, 1, 1, 1, 1, 127]),
            np.array([1, 1, 1, 1, 1, 1, 49], np.float32),
            [1, [1], 1.0, [1.0], np.array([1.0], dtype=np.float32), np.array([[1.0]], dtype=np.float32), "1"],
        ],
        [
            RLBaseTypes.NP_ARRAY,
            SpaceEncodeOptions(cast=False),
            NpArraySpace(7, 0, [1, 1, 1, 1, 1, 1, 127]),
            np.array([1, 1, 1, 1, 1, 1, 49], np.float32),
            [1, [1], 1.0, [1.0], np.array([1.0], dtype=np.float32), np.array([[1.0]], dtype=np.float32), "1"],
        ],
        [
            RLBaseTypes.BOX,
            None,
            BoxSpace((7,), 0, [1, 1, 1, 1, 1, 1, 127]),
            np.array([1, 1, 1, 1, 1, 1, 49], np.float32),
            [1, [1], 1.0, [1.0], np.array([1.0], dtype=np.float32), np.array([[1.0]], dtype=np.float32), "1"],
        ],
        [
            RLBaseTypes.BOX,
            SpaceEncodeOptions(cast=False),
            BoxSpace((7,), 0, [1, 1, 1, 1, 1, 1, 127]),
            np.array([1, 1, 1, 1, 1, 1, 49], np.float32),
            [1, [1], 1.0, [1.0], np.array([1.0], dtype=np.float32), np.array([[1.0]], dtype=np.float32), "1"],
        ],
        [
            RLBaseTypes.TEXT,
            None,
            TextSpace(-1, 1, "0123456789-,._"),
            "1_1_1.0_1.0_1.0_1.0_5",
            [1, [1], 1.0, [1.0], np.array([1.0], dtype=np.float32), np.array([[1.0]], dtype=np.float32), "5"],
        ],
    ],
)
def test_space_same_shape(create_space, options, true_space, val, decode_val):
    space = MultiSpace(
        [
            DiscreteSpace(2),
            ArrayDiscreteSpace(1, 0, 1),
            ContinuousSpace(0, 1),
            ArrayContinuousSpace(1, 0, 1),
            NpArraySpace(1, 0, 1),
            BoxSpace((1, 1), 0, 1),
            TextSpace(1, 1, "0123456789"),
        ]
    )
    print(space)
    assert space.stype == SpaceTypes.MULTI
    space.create_division_tbl(division_num=3)
    if options is None:
        options = SpaceEncodeOptions(cast=True)

    if true_space is None:
        with pytest.raises(NotSupportedError):
            space.create_encode_space(create_space, options)
        return

    target_space = space.create_encode_space(create_space, options)
    print(target_space)
    print(true_space)
    if create_space == RLBaseTypes.NONE:
        assert target_space == space
    else:
        assert target_space == true_space

    de = space.decode_from_space(val, target_space)
    for i in range(len(decode_val)):
        print(i, de[i], decode_val[i])
        if isinstance(de[i], np.ndarray):
            assert (de[i] == decode_val[i]).all()
        else:
            assert de[i] == decode_val[i]
    assert space.check_val(de)
    en = space.encode_to_space(decode_val, target_space)
    print(en)
    print(val)
    if isinstance(en, np.ndarray):
        assert (en == val).all()
    else:
        assert en == val
    assert target_space.check_val(en)

    de = space.decode_from_space(en, target_space)
    for i in range(len(decode_val)):
        print(i, de[i], decode_val[i])
        if isinstance(de[i], np.ndarray):
            assert (de[i] == decode_val[i]).all()
        else:
            assert de[i] == decode_val[i]
    assert space.check_val(de)


@pytest.mark.parametrize(
    "create_space, options, true_space, val, decode_val",
    [
        [
            RLBaseTypes.NONE,
            None,
            MultiSpace,
            [1, [0], 0.1, [0.5, 0.6, 0.7], np.array([0.2, 0.3], np.float32), np.ones((2, 1), np.float32), "5"],
            [1, [0], 0.1, [0.5, 0.6, 0.7], np.array([0.2, 0.3], np.float32), np.ones((2, 1), np.float32), "5"],
        ],
        [
            RLBaseTypes.DISCRETE,
            None,
            DiscreteSpace(152064, 0),
            0,
            [0, [0], 0.0, [0.0, 0.0, 0.0], np.array([0, 0], dtype=np.float32), np.array([[0], [0]], dtype=np.float32), "0"],
        ],
        [
            RLBaseTypes.ARRAY_DISCRETE,
            None,
            ArrayDiscreteSpace(8, 0, [1, 1, 3, 8, 4, 4, 127, 127]),
            [1, 1, 1, 1, 1, 1, 49, 49],
            [1, [1], 0.5, [0.0, 0.0, 1.0], np.array([0, 1], dtype=np.float32), np.array([[0], [1]], dtype=np.float32), "11"],
        ],
        [RLBaseTypes.CONTINUOUS, None, None, 1.1, 1.1],
        [
            RLBaseTypes.ARRAY_CONTINUOUS,
            None,
            ArrayContinuousSpace(12, 0, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 127, 127]),
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 49.0, 49.0],
            [1, [1], 1.0, [1.0, 1.0, 1.0], np.array([1, 1], dtype=np.float32), np.array([[1], [1]], dtype=np.float32), "11"],
        ],
        [
            RLBaseTypes.NP_ARRAY,
            None,
            NpArraySpace(12, 0, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 127, 127]),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 49, 49], np.float32),
            [1, [1], 1.0, [1.0, 1.0, 1.0], np.array([1, 1], dtype=np.float32), np.array([[1], [1]], dtype=np.float32), "11"],
        ],
        [
            RLBaseTypes.NP_ARRAY,
            SpaceEncodeOptions(cast=False),
            NpArraySpace(12, 0, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 127, 127]),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 49, 49], np.float32),
            [1, [1], 1.0, [1.0, 1.0, 1.0], np.array([1, 1], dtype=np.float32), np.array([[1], [1]], dtype=np.float32), "11"],
        ],
        [
            RLBaseTypes.BOX,
            None,
            BoxSpace((12,), 0, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 127, 127]),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 49, 49], np.float32),
            [1, [1], 1.0, [1.0, 1.0, 1.0], np.array([1, 1], dtype=np.float32), np.array([[1], [1]], dtype=np.float32), "11"],
        ],
        [
            RLBaseTypes.BOX,
            SpaceEncodeOptions(cast=False),
            BoxSpace((12,), 0, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 127, 127]),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 49, 49], np.float32),
            [1, [1], 1.0, [1.0, 1.0, 1.0], np.array([1, 1], dtype=np.float32), np.array([[1], [1]], dtype=np.float32), "11"],
        ],
        [
            RLBaseTypes.TEXT,
            None,
            TextSpace(-1, 1, "0123456789-,._"),
            "1_1_1.0_1.0,1.0,1.0_1.0,1.0_1.0,1.0_55",
            [1, [1], 1.0, [1.0, 1.0, 1.0], np.array([1, 1], dtype=np.float32), np.array([[1], [1]], dtype=np.float32), "55"],
        ],
    ],
)
def test_space_no_same_shape(create_space, options, true_space, val, decode_val):
    space = MultiSpace(
        [
            DiscreteSpace(2),
            ArrayDiscreteSpace(1, 0, 1),
            ContinuousSpace(0, 1),
            ArrayContinuousSpace(3, 0, 1),
            NpArraySpace(2, 0, 1),
            BoxSpace((2, 1), 0, 1),
            TextSpace(2, 1, "0123456789"),
        ]
    )
    print(space)
    assert space.stype == SpaceTypes.MULTI
    space.create_division_tbl(division_num=3)
    if options is None:
        options = SpaceEncodeOptions(cast=True)

    if true_space is None:
        with pytest.raises(NotSupportedError):
            space.create_encode_space(create_space, options)
        return

    target_space = space.create_encode_space(create_space, options)
    print(target_space)
    print(true_space)
    if create_space == RLBaseTypes.NONE:
        assert target_space == space
    else:
        assert target_space == true_space

    de = space.decode_from_space(val, target_space)
    for i in range(len(decode_val)):
        print(i, de[i], decode_val[i])
        if isinstance(de[i], np.ndarray):
            assert (de[i] == decode_val[i]).all()
        else:
            assert de[i] == decode_val[i]
    assert space.check_val(de)
    en = space.encode_to_space(decode_val, target_space)
    print(en)
    print(val)
    if create_space == RLBaseTypes.NONE:
        for i in range(len(val)):
            if isinstance(en[i], np.ndarray):
                assert (en[i] == val[i]).all()
            else:
                assert en[i] == val[i]
    else:
        if isinstance(en, np.ndarray):
            assert (en == val).all()
        else:
            assert en == val
    assert target_space.check_val(en)

    de = space.decode_from_space(en, target_space)
    for i in range(len(decode_val)):
        print(i, de[i], decode_val[i])
        if isinstance(de[i], np.ndarray):
            assert (de[i] == decode_val[i]).all()
        else:
            assert de[i] == decode_val[i]
    assert space.check_val(de)


@pytest.mark.parametrize(
    "create_space, options, true_space, val, decode_val",
    [
        [
            RLBaseTypes.BOX,
            None,
            BoxSpace((4, 3), 0, 1),
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], np.float32),
            [[1, 1, 1], [1.0, 1.0, 1.0], np.array([1, 1, 1], dtype=np.float32), np.array([1, 1, 1], dtype=np.float32)],
        ],
        [
            RLBaseTypes.BOX,
            SpaceEncodeOptions(cast=False),
            BoxSpace((4, 3), 0, 1),
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], np.float32),
            [[1, 1, 1], [1.0, 1.0, 1.0], np.array([1, 1, 1], dtype=np.float32), np.array([1, 1, 1], dtype=np.float32)],
        ],
    ],
)
def test_space_box(create_space, options, true_space, val, decode_val):
    space = MultiSpace(
        [
            ArrayDiscreteSpace(3, 0, 1),
            ArrayContinuousSpace(3, 0, 1),
            NpArraySpace(3, 0, 1),
            BoxSpace((3,), 0, 1),
        ]
    )
    print(space)
    assert space.stype == SpaceTypes.MULTI
    space.create_division_tbl(division_num=3)
    if options is None:
        options = SpaceEncodeOptions(cast=True)

    if true_space is None:
        with pytest.raises(NotSupportedError):
            space.create_encode_space(create_space, options)
        return

    target_space = space.create_encode_space(create_space, options)
    print(target_space)
    print(true_space)
    if create_space == RLBaseTypes.NONE:
        assert target_space == space
    else:
        assert target_space == true_space

    de = space.decode_from_space(val, target_space)
    for i in range(len(decode_val)):
        print(i, de[i], decode_val[i])
        if isinstance(de[i], np.ndarray):
            assert (de[i] == decode_val[i]).all()
        else:
            assert de[i] == decode_val[i]
    assert space.check_val(de)
    en = space.encode_to_space(decode_val, target_space)
    print(en)
    print(val)
    if isinstance(en, np.ndarray):
        assert (en == val).all()
    else:
        assert en == val
    assert target_space.check_val(en)

    de = space.decode_from_space(en, target_space)
    for i in range(len(decode_val)):
        print(i, de[i], decode_val[i])
        if isinstance(de[i], np.ndarray):
            assert (de[i] == decode_val[i]).all()
        else:
            assert de[i] == decode_val[i]
    assert space.check_val(de)
