import itertools

import numpy as np
import pytest

from srl.base.define import RLActionTypes
from srl.base.spaces import BoxSpace

from .space_test import SpaceTest


def _check_action(decode_action, true_shape, true_action):
    assert isinstance(decode_action, np.ndarray)
    assert "float" in str(decode_action.dtype)
    assert decode_action.shape == true_shape
    if true_action is not None:
        np.testing.assert_array_equal(decode_action, true_action)


def test_space1():
    space = BoxSpace((1,), -1, 3)
    tester = SpaceTest(space)
    assert space.rl_action_type == RLActionTypes.CONTINUOUS

    print(space)

    # sample
    for _ in range(100):
        action = space.sample()
        _check_action(action, (1,), None)
        assert np.min(action) >= -1
        assert np.max(action) <= 3

    # action discrete
    space.set_action_division(5)
    true_tbl = [
        [-1],
        [0],
        [1],
        [2],
        [3],
    ]
    assert space.action_tbl is not None
    np.testing.assert_array_equal(true_tbl[0], space.action_tbl[0])
    decode_action = tester.check_action_discrete(
        true_n=5,
        action=3,
    )
    _check_action(decode_action, (1,), [2])
    tester.check_action_encode(decode_action, 3)

    # action_continuous
    decode_action = tester.check_action_continuous(
        true_n=1,
        true_low=[-1],
        true_high=[3],
        action=[1.1],
    )
    _check_action(decode_action, (1,), [1.1])

    # observation discrete
    tester.check_observation_discrete(
        true_shape=(1,),
        state=np.array([1.1], dtype=np.float32),
        encode_state=[1],
    )

    # observation continuous
    tester.check_observation_continuous(
        true_shape=(1,),
        state=np.array([1.1], dtype=np.float32),
        encode_state=np.array([1.1], dtype=np.float32),
    )

    # eq
    assert space == BoxSpace((1,), -1, 3)
    assert space != BoxSpace((1,), -1, 2)


def test_space2():
    space = BoxSpace((3, 2), -1, 3)
    tester = SpaceTest(space)

    # sample
    for _ in range(100):
        action = space.sample()
        _check_action(action, (3, 2), None)
        assert np.min(action) >= -1
        assert np.max(action) <= 3

    # action discrete
    space.set_action_division(5)
    _t = list(itertools.product([-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]))
    true_tbl = list(itertools.product(_t, _t, _t))
    assert space.action_tbl is not None
    for a in range(len(true_tbl)):
        np.testing.assert_array_equal(true_tbl[a], space.action_tbl[a])
    decode_action = tester.check_action_discrete(
        true_n=5 ** (3 * 2),
        action=3,
    )
    _check_action(decode_action, (3, 2), true_tbl[3])
    tester.check_action_encode(decode_action, 3)

    # action_continuous
    decode_action = tester.check_action_continuous(
        true_n=6,
        true_low=[-1] * 6,
        true_high=[3] * 6,
        action=[1.1, 2.2, 3.0, -1.0, 1.5, 1.6],
    )
    _check_action(decode_action, (3, 2), [[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]])

    # observation discrete
    tester.check_observation_discrete(
        true_shape=(3, 2),
        state=np.array([[1.1, 2.2], [3.0, -1.0], [1.3, 1.6]], dtype=np.float32),
        encode_state=[[1, 2], [3, -1], [1, 2]],
    )

    # observation continuous
    tester.check_observation_continuous(
        true_shape=(3, 2),
        state=np.array([[1.1, 2.2], [3.0, -1.0], [1.3, 1.6]], dtype=np.float32),
        encode_state=np.array([[1.1, 2.2], [3.0, -1.0], [1.3, 1.6]], dtype=np.float32),
    )

    # eq
    assert space == BoxSpace((3, 2), -1, 3)
    assert space != BoxSpace((3, 3), -1, 3)


def test_inf():
    space = BoxSpace((3, 2))
    tester = SpaceTest(space)

    # sample
    for _ in range(100):
        action = space.sample()
        _check_action(action, (3, 2), None)

    # action discrete
    space.set_action_division(5)
    with pytest.raises(NotImplementedError):
        space.action_discrete_encode(None)  # type: ignore

    # action_continuous
    decode_action = tester.check_action_continuous(
        true_n=6,
        true_low=[-np.inf] * 6,
        true_high=[np.inf] * 6,
        action=[1.1, 2.2, 3.0, -1.0, 1.5, 1.6],
    )
    _check_action(decode_action, (3, 2), [[1.1, 2.2], [3.0, -1.0], [1.5, 1.6]])

    # observation discrete
    tester.check_observation_discrete(
        true_shape=(3, 2),
        state=np.array([[1.1, 2.2], [3.0, -1.0], [1.3, 1.6]], dtype=np.float32),
        encode_state=[[1, 2], [3, -1], [1, 2]],
    )

    # observation continuous
    tester.check_observation_continuous(
        true_shape=(3, 2),
        state=np.array([[1.1, 2.2], [3.0, -1.0], [1.3, 1.6]], dtype=np.float32),
        encode_state=np.array([[1.1, 2.2], [3.0, -1.0], [1.3, 1.6]], dtype=np.float32),
    )


def test_convert():
    space = BoxSpace((2,), -1, 3)

    val = space.convert([1.2, 2])
    assert space.check_val(val)
    np.testing.assert_array_equal([1.2, 2.0], val)

    val = space.convert([5, 2])
    assert not space.check_val(val)

    assert not space.check_val([1.2, 2.2])
    assert not space.check_val(np.array([1.2, 2.2, 2.2]))
    assert not space.check_val(np.array([1.2, -1.1]))
    assert not space.check_val(np.array([1.2, 3.1]))
