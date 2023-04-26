import numpy as np

from srl.base.define import RLActionType
from srl.base.env.spaces import DiscreteSpace

from .space_test import SpaceTest


def _check_action(decode_action, true_action):
    assert isinstance(decode_action, int)
    assert decode_action == true_action


def test_space():
    space = DiscreteSpace(5)
    assert space.base_action_type == RLActionType.DISCRETE
    assert space.n == 5

    print(space)

    tester = SpaceTest(space)

    # sample
    actions = [space.sample([3]) for _ in range(100)]
    actions = sorted(list(set(actions)))
    np.testing.assert_array_equal(actions, [0, 1, 2, 4])

    # action discrete
    decode_action = tester.check_action_discrete(5, action=2)
    _check_action(decode_action, 2)
    tester.check_action_encode(3, 3)

    # action_continuous
    decode_action = tester.check_action_continuous(
        true_n=1,
        true_low=[0],
        true_high=[4],
        action=[3.3],
    )
    _check_action(decode_action, 3)

    # observation discrete
    tester.check_observation_discrete(
        true_shape=(1,),
        state=2,
        encode_state=[2],
    )

    # observation continuous
    tester.check_observation_continuous(
        true_shape=(1,),
        state=2,
        encode_state=[2.0],
    )

    # eq
    assert space == DiscreteSpace(5)
    assert space != DiscreteSpace(4)


def test_convert():
    space = DiscreteSpace(5)

    val = space.convert([0.9])
    assert space.check_val(val)
    assert val == 1

    val = space.convert(0.9)
    assert space.check_val(val)
    assert val == 1

    val = space.convert(4)
    assert space.check_val(val)
    assert val == 4

    val = space.convert([0.9])
    assert space.check_val(val)
    assert val == 1

    val = space.convert((0.9,))
    assert space.check_val(val)
    assert val == 1

    assert not space.check_val(1.1)
    assert not space.check_val(-1)
    assert not space.check_val(5)
