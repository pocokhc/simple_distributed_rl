import numpy as np

import srl
from srl.base.define import DoneTypes
from srl.test.env import TestEnv


def test_play():
    tester = TestEnv()
    tester.play_test("OX")


def test_player():
    tester = TestEnv()
    tester.player_test("OX", "cpu")


def test_processor():
    env = srl.make_env(srl.EnvConfig("OX", {"obs_type": "layer"}))
    env.reset()

    out_state = np.zeros((3, 3, 2))

    assert (out_state == env.state).all()


def test_play_step():
    env = srl.make_env("OX")

    env.reset()
    np.testing.assert_array_equal(env.state, [0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert env.next_player_index == 0

    # 1
    env.step(0)
    assert env.step_num == 1
    assert env.next_player_index == 1
    np.testing.assert_array_equal(env.state, [1, 0, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(env.get_invalid_actions(), [0])
    np.testing.assert_array_equal(env.step_rewards, [0, 0])
    assert not env.done

    # 2
    env.step(1)
    assert env.step_num == 2
    assert env.next_player_index == 0
    np.testing.assert_array_equal(env.state, [1, -1, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(env.get_invalid_actions(), [0, 1])
    np.testing.assert_array_equal(env.step_rewards, [0, 0])
    assert not env.done

    # 3
    env.step(2)
    assert env.step_num == 3
    assert env.next_player_index == 1
    np.testing.assert_array_equal(env.state, [1, -1, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(env.get_invalid_actions(), [0, 1, 2])
    np.testing.assert_array_equal(env.step_rewards, [0, 0])
    assert not env.done

    # 4
    env.step(3)
    assert env.step_num == 4
    assert env.next_player_index == 0
    np.testing.assert_array_equal(env.state, [1, -1, 1, -1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(env.get_invalid_actions(), [0, 1, 2, 3])
    np.testing.assert_array_equal(env.step_rewards, [0, 0])
    assert not env.done

    # 5
    env.step(4)
    assert env.step_num == 5
    assert env.next_player_index == 1
    np.testing.assert_array_equal(env.state, [1, -1, 1, -1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(env.get_invalid_actions(), [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(env.step_rewards, [0, 0])
    assert not env.done

    # 6
    env.step(5)
    assert env.step_num == 6
    assert env.next_player_index == 0
    np.testing.assert_array_equal(env.state, [1, -1, 1, -1, 1, -1, 0, 0, 0])
    np.testing.assert_array_equal(env.get_invalid_actions(), [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(env.step_rewards, [0, 0])
    assert not env.done

    # 7
    env.step(7)
    assert env.step_num == 7
    assert env.next_player_index == 1
    np.testing.assert_array_equal(env.state, [1, -1, 1, -1, 1, -1, 0, 1, 0])
    np.testing.assert_array_equal(env.get_invalid_actions(), [0, 1, 2, 3, 4, 5, 7])
    np.testing.assert_array_equal(env.step_rewards, [0, 0])
    assert not env.done

    # 8
    env.step(6)
    assert env.step_num == 8
    assert env.next_player_index == 0
    np.testing.assert_array_equal(env.state, [1, -1, 1, -1, 1, -1, -1, 1, 0])
    np.testing.assert_array_equal(env.get_invalid_actions(), [0, 1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(env.step_rewards, [0, 0])
    assert not env.done

    # 9
    env.step(8)
    assert env.step_num == 9
    assert env.next_player_index == 0
    np.testing.assert_array_equal(env.state, [1, -1, 1, -1, 1, -1, -1, 1, 1])
    np.testing.assert_array_equal(env.get_invalid_actions(), [0, 1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(env.step_rewards, [1, -1])
    assert env.done
    assert env.done_type == DoneTypes.TERMINATED
    assert env.done_reason == ""
