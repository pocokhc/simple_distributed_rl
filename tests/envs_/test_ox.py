import numpy as np

import srl
from srl.base.define import DoneTypes, EnvObservationTypes
from srl.base.spaces.box import BoxSpace
from srl.envs import ox  # noqa F401
from srl.test import TestEnv
from srl.test.processor import TestProcessor


def test_play():
    tester = TestEnv()
    tester.play_test("OX")


def test_player():
    tester = TestEnv()
    tester.player_test("OX", "cpu")


def test_processor():
    tester = TestProcessor()
    processor = ox.LayerProcessor()
    env_name = "OX"

    in_state = [0] * 9
    out_state = np.zeros((3, 3, 2))

    tester.run(processor, env_name)
    tester.preprocess_observation_space(
        processor,
        env_name,
        EnvObservationTypes.IMAGE,
        BoxSpace((3, 3, 2), 0, 1),
    )
    tester.preprocess_observation(
        processor,
        env_name,
        in_observation=in_state,
        out_observation=out_state,
    )


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
    assert env.done_reason == DoneTypes.TERMINATED
