import numpy as np
import pytest

from srl.base.define import RLActionType
from srl.base.env.spaces import ContinuousSpace
from tests.base.env.space_test import SpaceTest


def _check_action(decode_action, true_action):
    assert isinstance(decode_action, float)
    if true_action is not None:
        assert decode_action == true_action


def test_space():
    space = ContinuousSpace(-1, 3)
    tester = SpaceTest(space)
    assert space.base_action_type == RLActionType.CONTINUOUS

    # sample
    for _ in range(100):
        action = space.sample()
        _check_action(action, None)
        assert action >= -1
        assert action <= 3

    # action discrete
    space.set_action_division(5)
    true_tbl = [
        [-1],
        [0],
        [1],
        [2],
        [3],
    ]
    np.testing.assert_array_equal(true_tbl[0], space.action_tbl[0])
    decode_action = tester.check_action_discrete(
        true_n=5,
        action=3,
    )
    _check_action(decode_action, 2)
    tester.check_action_encode(decode_action, 3)

    # action_continuous
    decode_action = tester.check_action_continuous(
        true_n=1,
        true_low=[-1],
        true_high=[3],
        action=[1.1],
    )
    _check_action(decode_action, 1.1)

    # observation discrete
    tester.check_observation_discrete(
        true_shape=(1,),
        state=1.1,
        encode_state=[1],
    )

    # observation continuous
    tester.check_observation_continuous(
        true_shape=(1,),
        state=1.1,
        encode_state=np.array([1.1], dtype=np.float32),
    )

    # eq
    assert space == ContinuousSpace(-1, 3)
    assert space != ContinuousSpace(-1, 2)


def test_inf():
    space = ContinuousSpace()
    tester = SpaceTest(space)

    print(space)

    # sample
    for _ in range(100):
        action = space.sample()
        _check_action(action, None)

    # action discrete
    space.set_action_division(5)
    decode_action = tester.check_action_discrete(
        true_n=0,
        action=3,
    )
    _check_action(decode_action, 3)
    with pytest.raises(NotImplementedError):
        space.action_discrete_encode(decode_action)

    # action_continuous
    decode_action = tester.check_action_continuous(
        true_n=1,
        true_low=[-np.inf],
        true_high=[np.inf],
        action=[1.1],
    )
    _check_action(decode_action, 1.1)

    # observation discrete
    tester.check_observation_discrete(
        true_shape=(1,),
        state=1.1,
        encode_state=[1],
    )

    # observation continuous
    tester.check_observation_continuous(
        true_shape=(1,),
        state=1.1,
        encode_state=np.array([1.1], dtype=np.float32),
    )


def test_convert():
    space = ContinuousSpace(-1, 3)

    val = space.convert(1)
    assert space.check_val(val)
    assert val == 1.0

    val = space.convert([2])
    assert space.check_val(val)
    assert val == 2.0

    val = space.convert((1,))
    assert space.check_val(val)
    assert val == 1.0

    val = space.convert(-2)
    assert not space.check_val(val)

    assert not space.check_val(1)
    assert not space.check_val(3.1)
    assert not space.check_val(-1.1)
