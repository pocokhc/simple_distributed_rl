import numpy as np
import pytest

from srl.base import spaces
from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.np_array import NpArraySpace
from srl.base.spaces.space import SpaceEncodeOptions
from srl.base.spaces.text import TextSpace


def test_space_basic():
    space = BoxSpace((3, 1), -1, 3)

    assert space.shape == (3, 1)
    np.testing.assert_array_equal(space.low, np.full((3, 1), -1))
    np.testing.assert_array_equal(space.high, np.full((3, 1), 3))
    assert space.stype == SpaceTypes.CONTINUOUS
    assert space.dtype == np.float32

    # --- check_val
    assert space.check_val(np.full((3, 1), 0))
    assert not space.check_val(np.full((3, 1), -2))

    # --- to_str
    assert space.to_str(np.full((3, 1), 0)) == "0,0,0"

    # --- copy
    c = space.copy()
    assert space == c

    # --- get_default
    assert (space.get_default() == np.zeros((3, 1))).all()


def test_stack():
    space = BoxSpace((3, 1), -1, 3)

    o = space.create_stack_space(2)
    assert isinstance(o, BoxSpace)
    assert o == BoxSpace((2, 3, 1), -1, 3)

    v = space.encode_stack([np.ones((3, 1)), np.ones((3, 1))])
    assert v.shape == (2, 3, 1)

    # --- gray
    space = BoxSpace((64, 32), 0, 255, stype=SpaceTypes.GRAY_2ch)

    o = space.create_stack_space(2)
    assert isinstance(o, BoxSpace)
    assert o == BoxSpace((64, 32, 2), 0, 255, stype=SpaceTypes.IMAGE)

    v = space.encode_stack([np.ones((64, 32)), np.ones((64, 32))])
    assert v.shape == (64, 32, 2)


def test_space_type():
    space = BoxSpace((2, 3), 0, 3, dtype=np.int8)
    assert space.stype == SpaceTypes.DISCRETE
    assert space.dtype == np.int8
    assert space != BoxSpace((2, 3), 0, 3, dtype=np.uint8)

    space = BoxSpace((2, 3), -1, 3, stype=SpaceTypes.COLOR)
    assert space.stype == SpaceTypes.COLOR
    assert space.dtype == np.float32
    assert space != BoxSpace((2, 3), -1, 3, stype=SpaceTypes.GRAY_3ch)


def test_discrete():
    space = BoxSpace((3, 1), 0, np.array([[2], [5], [3]], np.int64), np.int64)
    print(space)
    assert space.stype == SpaceTypes.DISCRETE

    # --- discrete
    enc_space = space.set_encode_space(RLBaseTypes.DISCRETE)
    assert enc_space.n == (2 + 1) * (5 + 1) * (3 + 1)
    de = space.decode_from_space(1)
    assert isinstance(de, np.ndarray)
    assert de.shape == (3, 1)
    np.testing.assert_array_equal(de, [[0], [0], [1]])
    en = space.encode_to_space(np.array([[0], [0], [1]], np.int64))
    assert isinstance(en, int)
    assert en == 1

    # --- list int
    enc_space = space.set_encode_space(RLBaseTypes.ARRAY_DISCRETE)
    assert enc_space.size == 3
    assert enc_space.low == [0, 0, 0]
    assert enc_space.high == [2, 5, 3]
    de = space.decode_from_space([1, 2, 0])
    assert isinstance(de, np.ndarray)
    assert de.shape == (3, 1)
    np.testing.assert_array_equal(de, [[1], [2], [0]])
    en = space.encode_to_space(np.array([[1], [2], [0]], np.int64))
    assert isinstance(en, list)
    assert len(en) == 3
    np.testing.assert_array_equal(en, [1, 2, 0])

    # --- list float
    enc_space = space.set_encode_space(RLBaseTypes.ARRAY_CONTINUOUS)
    assert enc_space.size == 3
    np.testing.assert_array_equal(enc_space.low, [0, 0, 0])
    np.testing.assert_array_equal(enc_space.high, [2, 5, 3])
    de = space.decode_from_space([0.1, 0.6, 1.9])
    assert isinstance(de, np.ndarray)
    assert de.shape == (3, 1)
    np.testing.assert_array_equal(de, [[0], [0], [1]])
    en = space.encode_to_space(np.array([[0], [1], [1]], np.int64))
    assert isinstance(en, list)
    for n in en:
        assert isinstance(n, float)
    np.testing.assert_array_equal(en, [0.0, 1.0, 1.0])

    # --- continuous numpy
    enc_space = space.set_encode_space(RLBaseTypes.BOX, SpaceEncodeOptions(cast=True, cast_dtype=np.int64))
    assert enc_space.shape == (3, 1)
    np.testing.assert_array_equal(enc_space.low, [[0], [0], [0]])
    np.testing.assert_array_equal(enc_space.high, [[2], [5], [3]])
    de = space.decode_from_space(np.array([[0.1], [0.6], [1.9]]))
    assert isinstance(de, np.ndarray)
    assert de.shape == (3, 1)
    np.testing.assert_array_equal(de, [[0], [0], [1]])
    en = space.encode_to_space(np.array([[0], [1], [1]], np.int64))
    np.testing.assert_array_equal(en, np.array([[0], [1], [1]], np.int64))

    # --- sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, np.ndarray)
        assert action.shape == (3, 1)
        assert 0 <= action[0][0] <= 2
        assert 0 <= action[1][0] <= 5
        assert 0 <= action[2][0] <= 3
        print(action)

    # --- eq
    assert space == BoxSpace((3, 1), 0, np.array([[2], [5], [3]], np.int64), np.int64)
    assert space != BoxSpace((3, 1), 0, np.array([[3], [5], [3]], np.int64), np.int64)


def test_con_no_division():
    space = BoxSpace((3, 2), -1, 3)
    print(space)

    # --- discrete
    with pytest.raises(AssertionError):
        space.set_encode_space(RLBaseTypes.DISCRETE)

    # --- list int
    enc_space = space.set_encode_space(RLBaseTypes.ARRAY_DISCRETE)
    assert enc_space.size == 6
    assert enc_space.low == [-1] * 6
    assert enc_space.high == [3] * 6
    en = space.encode_to_space(np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]]))
    assert len(en) == 6
    assert en == [1, 2, 3, -1, 2, 2]
    de = space.decode_from_space([1, 2, 3, -1, 1, 1])
    assert de.shape == (3, 2)
    np.testing.assert_array_equal(de, np.array([[1, 2], [3, -1], [1, 1]], np.float32))

    # --- list float
    enc_space = space.set_encode_space(RLBaseTypes.ARRAY_CONTINUOUS)
    assert enc_space.size == 6
    np.testing.assert_array_equal(enc_space.low, [-1] * 6)
    np.testing.assert_array_equal(enc_space.high, [3] * 6)
    de = space.decode_from_space(np.array([1.1, 2.2, 3.0, -1.0, 1.5, 1.6]))
    np.testing.assert_array_equal(de, np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]], np.float32))
    en = space.encode_to_space(np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]]))
    assert isinstance(en, list)
    for n in en:
        assert isinstance(n, float)
    np.testing.assert_array_equal(en, [1.1, 2.2, 3.0, -1.0, 1.5, 1.6])

    # --- continuous numpy
    enc_space = space.set_encode_space(RLBaseTypes.BOX)
    assert enc_space.shape == (3, 2)
    np.testing.assert_array_equal(enc_space.low, np.array([-1] * 6).reshape((3, 2)))
    np.testing.assert_array_equal(enc_space.high, np.array([3] * 6).reshape((3, 2)))
    de = space.decode_from_space(np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]], np.float32))
    np.testing.assert_array_equal(de, np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]], np.float32))
    en = space.encode_to_space(np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]], np.float32))
    np.testing.assert_array_equal(en, np.array([[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]], dtype=np.float32))

    # --- sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, np.ndarray)
        assert "float" in str(action.dtype)
        assert action.shape == (3, 2)
        assert np.min(action) >= -1
        assert np.max(action) <= 3

    # --- eq
    assert space == BoxSpace((3, 2), -1, 3)
    assert space != BoxSpace((3, 2), -1, 2)
    assert space != BoxSpace((3, 3), -1, 3)


def test_con_division():
    space = BoxSpace((3, 2), -1, 3)

    # action discrete
    space.create_division_tbl(100)
    assert space.division_tbl is not None

    # --- discrete
    enc_space = space.set_encode_space(RLBaseTypes.DISCRETE)
    assert enc_space.n == 2 ** (3 * 2)
    en = space.encode_to_space(np.array([[-0.9, -1], [-1, -0.9], [-1, -1]]))
    assert en == 0
    de = space.decode_from_space(0)
    np.testing.assert_array_equal(de, [[-1, -1], [-1, -1], [-1, -1]])

    # --- list int
    enc_space = space.set_encode_space(RLBaseTypes.ARRAY_DISCRETE)
    en = space.encode_to_space(np.array([[-0.9, -1], [-1, -0.9], [-1, -1]]))
    assert len(en) == 1
    assert en[0] == 0
    de = space.decode_from_space([0])
    np.testing.assert_array_equal(de, [[-1, -1], [-1, -1], [-1, -1]])


def test_inf():
    space = BoxSpace((3, 2))

    # sample
    for _ in range(100):
        action = space.sample()
        assert isinstance(action, np.ndarray)
        assert "float" in str(action.dtype)
        assert action.shape == (3, 2)

    # --- discrete
    space.create_division_tbl(5)
    with pytest.raises(AssertionError):
        space.set_encode_space(RLBaseTypes.DISCRETE)

    # --- list float
    enc_space = space.set_encode_space(RLBaseTypes.ARRAY_CONTINUOUS)
    assert enc_space.size == 6
    np.testing.assert_array_equal(enc_space.low, [-np.inf] * 6)
    np.testing.assert_array_equal(enc_space.high, [np.inf] * 6)


def test_rescale_from():
    space = BoxSpace((3,), 0, 1)
    x = np.array([10, 11, 12])
    y = space.rescale_from(x, 10, 12)
    assert (y == np.array([0, 0.5, 1], dtype=np.float32)).all()


def test_sanitize():
    space = BoxSpace((2, 1), -1, 3)

    val = space.sanitize([[1.2], [2]])
    assert space.check_val(val)
    np.testing.assert_array_equal(np.array([[1.2], [2.0]], dtype=np.float32), val)

    assert not space.check_val(np.array([1.2, 2.2], dtype=np.float32))
    assert not space.check_val([[1.2], [2.2]])
    assert not space.check_val(np.array([[1.2], [2.2], [2.2]], dtype=np.float32))
    assert not space.check_val(np.array([[1.2], [-1.1]], dtype=np.float32))
    assert not space.check_val(np.array([[1.2], [3.1]], dtype=np.float32))


def test_sanitize2():
    space = BoxSpace((2, 1), -1, 3)

    val = space.sanitize([[-2], [6]])
    assert space.check_val(val)
    np.testing.assert_array_equal([[-1], [3]], val)


def test_sample1():
    space = BoxSpace((1, 1), -1, 0, np.int64)
    for _ in range(100):
        r = space.sample().tolist()[0][0]
        assert isinstance(r, int)
        assert r in [-1, 0]


def test_sample2():
    space = BoxSpace((2, 1), -1, 0, np.int64)
    for _ in range(100):
        r = space.sample(
            [
                np.array([[0], [0]], np.int64),
                np.array([[-1], [0]], np.int64),
                np.array([[0], [-1]], np.int64),
            ]
        )
        assert r.tolist() == [[-1], [-1]]


def test_valid_actions():
    space = BoxSpace((2, 1), -1, 0, np.int64)
    acts = space.get_valid_actions(
        [
            np.array([[-1], [-1]], np.int64),
            np.array([[-1], [0]], np.int64),
            np.array([[0], [-1]], np.int64),
        ]
    )
    assert len(acts) == 1
    assert (acts[0] == [[0], [0]]).all()


DB = BoxSpace((2, 1), -1, 0, np.int64)
CB = BoxSpace((2, 1), -1, 0)


@pytest.mark.parametrize(
    "space, create_space, options, true_space, val, decode_val",
    [
        [DB, RLBaseTypes.NONE, None, BoxSpace((2, 1), -1, 0, np.int64), np.array([[-1], [-1]]), [[-1], [-1]]],
        [DB, RLBaseTypes.DISCRETE, None, spaces.DiscreteSpace(4), 0, [[-1], [-1]]],
        [DB, RLBaseTypes.ARRAY_DISCRETE, None, spaces.ArrayDiscreteSpace(2, -1, 0), [-1, -1], [[-1], [-1]]],
        [DB, RLBaseTypes.CONTINUOUS, None, None, 0, [[-1], [-1]]],
        [DB, RLBaseTypes.ARRAY_CONTINUOUS, None, spaces.ArrayContinuousSpace(2, -1, 0), [-1, -1], [[-1], [-1]]],
        [DB, RLBaseTypes.NP_ARRAY, None, NpArraySpace(2, -1, 0, stype=SpaceTypes.DISCRETE), np.array([-1, -1], np.float32), [[-1], [-1]]],
        [DB, RLBaseTypes.NP_ARRAY, SpaceEncodeOptions(cast=False), NpArraySpace(2, -1, 0, np.int64), np.array([-1, -1], np.int64), [[-1], [-1]]],
        # box
        [DB, RLBaseTypes.BOX, None, BoxSpace((2, 1), -1, 0, np.float32, SpaceTypes.DISCRETE), np.array([[-1], [-1]], np.float32), [[-1], [-1]]],
        [DB, RLBaseTypes.BOX, SpaceEncodeOptions(cast=False), BoxSpace((2, 1), -1, 0, np.int64), np.array([[-1], [-1]]), [[-1], [-1]]],
        [CB, RLBaseTypes.BOX, None, BoxSpace((2, 1), -1.0, 0.0, np.float32), np.array([[-1], [-1]], np.float32), [[-1], [-1]]],
        [CB, RLBaseTypes.BOX, SpaceEncodeOptions(cast=False), BoxSpace((2, 1), -1, 0, np.float32), np.array([[-1], [-1]], np.float32), [[-1], [-1]]],
        # image
        [
            BoxSpace((8, 4), 0, 255, np.uint, SpaceTypes.GRAY_2ch),
            RLBaseTypes.BOX,
            None,
            BoxSpace((8, 4), 0, 255, np.float32, SpaceTypes.GRAY_2ch),
            np.full((8, 4), 2),
            np.full((8, 4), 2),
        ],
        [
            BoxSpace((8, 4, 1), 0, 255, np.uint, SpaceTypes.GRAY_3ch),
            RLBaseTypes.BOX,
            None,
            BoxSpace((8, 4, 1), 0, 255, np.float32, SpaceTypes.GRAY_3ch),
            np.full((8, 4, 1), 2),
            np.full((8, 4, 1), 2),
        ],
        [
            BoxSpace((8, 4, 3), 0, 255, np.uint, SpaceTypes.COLOR),
            RLBaseTypes.BOX,
            None,
            BoxSpace((8, 4, 3), 0, 255, np.float32, SpaceTypes.COLOR),
            np.full((8, 4, 3), 2),
            np.full((8, 4, 3), 2),
        ],
        [
            BoxSpace((8, 4, 2), 0, 255, np.int64, SpaceTypes.IMAGE),
            RLBaseTypes.BOX,
            None,
            BoxSpace((8, 4, 2), 0, 255, np.float32, SpaceTypes.IMAGE),
            np.full((8, 4, 2), 2),
            np.full((8, 4, 2), 2),
        ],
        [
            CB,
            RLBaseTypes.TEXT,
            None,
            TextSpace(min_length=1, charset="0123456789-.,"),
            "-1.0,0.0",
            np.array([[-1.0], [0.0]], np.float32),
        ],
    ],
)
def test_space(space: BoxSpace, create_space, options, true_space, val, decode_val):
    decode_val = np.array(decode_val)
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
        [RLBaseTypes.NP_ARRAY, SpaceEncodeOptions(np_zero_start=True), NpArraySpace(2, 0, 2, np.int64), np.full((2, 1), 2, np.int64), 5],
        [RLBaseTypes.NP_ARRAY, SpaceEncodeOptions(np_norm_type="0to1"), NpArraySpace(2, 0, 1, np.float32), np.full((2, 1), 1, np.float32), 5],
        [RLBaseTypes.NP_ARRAY, SpaceEncodeOptions(np_norm_type="-1to1"), NpArraySpace(2, -1, 1, np.float32), np.full((2, 1), -1, np.float32), 3],
        [RLBaseTypes.BOX, SpaceEncodeOptions(np_zero_start=True), BoxSpace((2, 1), 0, 2, np.int64), np.full((2, 1), 2, np.float32), 5],
        [RLBaseTypes.BOX, SpaceEncodeOptions(np_norm_type="0to1"), BoxSpace((2, 1), 0, 1, np.float32), np.full((2, 1), 1, np.float32), 5],
        [RLBaseTypes.BOX, SpaceEncodeOptions(np_norm_type="-1to1"), BoxSpace((2, 1), -1, 1, np.float32), np.full((2, 1), -1, np.float32), 3],
    ],
)
def test_space_zero_start(create_space, options, true_space, val, decode_val):
    decode_val = np.full((2, 1), decode_val, np.int64)
    space = BoxSpace((2, 1), 3, 5, np.int64)
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
