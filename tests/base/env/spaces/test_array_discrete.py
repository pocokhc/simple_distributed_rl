import numpy as np

from srl.base.define import RLActionType
from srl.base.env.spaces import ArrayDiscreteSpace
from tests.base.env.space_test import SpaceTest


def _check_action(decode_action, size, true_action):
    assert isinstance(decode_action, list)
    assert len(decode_action) == size
    for a in decode_action:
        assert isinstance(a, int)
    if true_action is not None:
        np.testing.assert_array_equal(decode_action, true_action)


def test_space():
    space = ArrayDiscreteSpace(3, 0, [2, 5, 3])
    assert space.base_action_type == RLActionType.DISCRETE
    assert space.size, 3
    np.testing.assert_array_equal(space.low, [0, 0, 0])
    np.testing.assert_array_equal(space.high, [2, 5, 3])

    tester = SpaceTest(space)

    print(space)

    # sample
    for _ in range(100):
        action = space.sample()
        _check_action(action, 3, None)
        assert 0 <= action[0] <= 2
        assert 0 <= action[1] <= 5
        assert 0 <= action[2] <= 3

    # action discrete
    decode_action = tester.check_action_discrete(
        true_n=(2 + 1) * (5 + 1) * (3 + 1),
        action=1,
    )
    _check_action(decode_action, 3, [0, 0, 1])
    tester.check_action_encode([0, 0, 1], 1)

    # action_continuous
    decode_action = tester.check_action_continuous(
        true_n=3,
        true_low=[0, 0, 0],
        true_high=[2, 5, 3],
        action=[0.1, 0.6, 0.9],
    )
    _check_action(decode_action, 3, [0, 1, 1])

    # observation discrete
    tester.check_observation_discrete(
        true_shape=(3,),
        state=[0, 0, 1],
        encode_state=[0, 0, 1],
    )

    # observation continuous
    tester.check_observation_continuous(
        true_shape=(3,),
        state=[0, 0, 1],
        encode_state=[0, 0, 1],
    )

    # eq
    assert space == ArrayDiscreteSpace(3, 0, [2, 5, 3])
    assert space != ArrayDiscreteSpace(3, 0, [3, 5, 3])


def test_convert():
    space = ArrayDiscreteSpace(3, 0, [2, 5, 3])

    val = space.convert(1.2)
    assert space.check_val(val)
    np.testing.assert_array_equal([1, 1, 1], val)

    val = space.convert([1.2, 0.9, 1.1])
    assert space.check_val(val)
    np.testing.assert_array_equal([1, 1, 1], val)

    val = space.convert((0, 4, True))
    assert space.check_val(val)
    np.testing.assert_array_equal([0, 4, 1], val)

    val = space.convert(np.array([1.2, 0.9, 1.1]))
    assert space.check_val(val)
    np.testing.assert_array_equal([1, 1, 1], val)

    val = space.convert(10)
    assert not space.check_val(val)

    assert not space.check_val(1)
    assert not space.check_val([1, 2])
    assert not space.check_val([1, 1, 1.1])
    assert not space.check_val([-1, 1, 1])
    assert not space.check_val([2, 5, 4])
