import numpy as np
import pytest

from srl.base.define import RLActionType
from srl.base.env.spaces import ArrayContinuousSpace
from tests.base.env.space_test import SpaceTest


def _check_action(decode_action, size, true_action):
    assert isinstance(decode_action, list)
    assert len(decode_action) == size
    for a in decode_action:
        assert isinstance(a, float)
    if true_action is not None:
        np.testing.assert_array_equal(decode_action, true_action)


def test_space():
    space = ArrayContinuousSpace(3, -1, 3)
    tester = SpaceTest(space)
    assert space.base_action_type == RLActionType.CONTINUOUS

    print(space)

    # sample
    for _ in range(100):
        action = space.sample()
        _check_action(action, 3, None)
        assert -1 <= action[0] <= 3
        assert -1 <= action[1] <= 3
        assert -1 <= action[2] <= 3

    # action discrete
    space.set_action_division(5)
    decode_action = tester.check_action_discrete(
        true_n=5**3,
        action=3,
    )
    _check_action(decode_action, 3, [-1, -1, 2])
    tester.check_action_encode(decode_action, 3)

    # action_continuous
    decode_action = tester.check_action_continuous(
        true_n=3,
        true_low=[-1] * 3,
        true_high=[3] * 3,
        action=[1.1, 0.1, 0.9],
    )
    _check_action(decode_action, 3, [1.1, 0.1, 0.9])

    # observation discrete
    tester.check_observation_discrete(
        true_shape=(3,),
        state=[1.1, 0.1, 0.9],
        encode_state=[1, 0, 1],
    )

    # observation continuous
    tester.check_observation_continuous(
        true_shape=(3,),
        state=[1.1, 0.1, 0.9],
        encode_state=np.array([1.1, 0.1, 0.9], dtype=np.float32),
    )


def test_inf():
    space = ArrayContinuousSpace(3)
    tester = SpaceTest(space)

    # sample
    for _ in range(100):
        action = space.sample()
        _check_action(action, 3, None)

    # action discrete
    space.set_action_division(5)
    decode_action = tester.check_action_discrete(
        true_n=0,
        action=3,
    )
    _check_action(decode_action, 3, [3, 3, 3])
    with pytest.raises(NotImplementedError):
        space.action_discrete_encode(decode_action)

    # action_continuous
    decode_action = tester.check_action_continuous(
        true_n=3,
        true_low=[-np.inf] * 3,
        true_high=[np.inf] * 3,
        action=[1.1, 0.1, 0.9],
    )
    _check_action(decode_action, 3, [1.1, 0.1, 0.9])

    # observation discrete
    tester.check_observation_discrete(
        true_shape=(3,),
        state=[1.1, 0.1, 0.9],
        encode_state=[1, 0, 1],
    )

    # observation continuous
    tester.check_observation_continuous(
        true_shape=(3,),
        state=[1.1, 0.1, 0.9],
        encode_state=np.array([1.1, 0.1, 0.9], dtype=np.float32),
    )


def test_convert():
    space = ArrayContinuousSpace(3, -1, 3)

    val = space.convert(1)
    assert space.check_val(val)
    np.testing.assert_array_equal([1.0, 1.0, 1.0], val)

    val = space.convert([1.2, 0.9, 0.8])
    assert space.check_val(val)
    np.testing.assert_array_equal([1.2, 0.9, 0.8], val)

    val = space.convert((2, 1, True))
    assert space.check_val(val)
    np.testing.assert_array_equal([2.0, 1.0, 1.0], val)

    val = space.convert(np.array([1.2, 0.9, 0.8]))
    assert space.check_val(val)
    np.testing.assert_array_equal([1.2, 0.9, 0.8], val)

    val = space.convert(5)
    assert not space.check_val(val)

    assert not space.check_val(1)
    assert not space.check_val([1])
    assert not space.check_val([1.1, 1.1, 1])
    assert not space.check_val([-2.1, 1.1, 1.1])
    assert not space.check_val([5.1, 1.1, 1.1])
