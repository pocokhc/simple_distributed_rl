from typing import cast

import numpy as np

import srl
from srl.envs import othello  # noqa F401
from srl.test.env import env_test, player_test


def test_play():
    env_test("Othello")


def test_play6x6():
    env_test("Othello6x6")


def test_play4x4():
    env_test("Othello4x4")


def test_player():
    player_test("Othello", "cpu", timeout=2)


def test_player6x6():
    player_test("Othello6x6", "cpu", timeout=1)


def test_player4x4():
    player_test("Othello4x4", "cpu")


def test_layer():
    env = srl.make_env(srl.EnvConfig("Othello-layer"))
    env.setup()
    env.reset()

    out_state = np.zeros((8, 8, 2))
    out_state[3][3][0] = 1
    out_state[4][4][0] = 1
    out_state[3][4][1] = 1
    out_state[4][3][1] = 1

    assert (out_state == env.state).all()


# ------------------------
def test_othello():
    env_run = srl.make_env("Othello")
    env = cast(othello.Othello, env_run.unwrapped)

    env_run.setup(render_mode="terminal")
    env_run.reset()
    env_run.render()

    # common func
    assert env.pos(3, 5) == 43
    assert env.pos_decode(43) == (3, 5)

    """
    | 0| 1| 2| 3| 4| 5| 6| 7|
    | 8| 9|10|11|12|13|14|15|
    |16|17|18|19|20|21|22|23|
    |24|25|26| o| x|29|30|31|
    |32|33|34| x| o|37|38|39|
    |40|41|42|43|44|45|46|47|
    |48|49|50|51|52|53|54|55|
    |56|57|58|59|60|61|62|63|
    """

    # 初期配置で置ける場所
    assert set(env.movable_dirs[0][34]) == set([6])
    assert set(env.movable_dirs[0][43]) == set([8])
    assert set(env.movable_dirs[0][20]) == set([2])
    assert set(env.movable_dirs[0][29]) == set([4])
    assert set(env.movable_dirs[1][19]) == set([2])
    assert set(env.movable_dirs[1][26]) == set([6])
    assert set(env.movable_dirs[1][37]) == set([4])
    assert set(env.movable_dirs[1][44]) == set([8])

    env_run.step(34)
    env_run.render()
    assert env.get_field(1, 4) == 0
    assert env.get_field(2, 4) == 1
    assert env.get_field(3, 4) == 1
    assert env.get_field(4, 4) == 1
    assert env.get_field(5, 4) == 0
    assert set(env.movable_dirs[0][20]) == set([2])
    assert set(env.movable_dirs[0][21]) == set([1])
    assert set(env.movable_dirs[0][29]) == set([4])
    assert set(env.movable_dirs[1][26]) == set([6])
    assert set(env.movable_dirs[1][42]) == set([9])
    assert set(env.movable_dirs[1][44]) == set([8])

    env_run.step(26)
    env_run.render()
    assert env.get_field(1, 3) == 0
    assert env.get_field(2, 3) == -1
    assert env.get_field(3, 3) == -1
    assert env.get_field(4, 3) == -1
    assert env.get_field(5, 3) == 0
    assert set(env.movable_dirs[0][17]) == set([3])
    assert set(env.movable_dirs[0][18]) == set([2, 3])
    assert set(env.movable_dirs[0][19]) == set([2])
    assert set(env.movable_dirs[0][20]) == set([1, 2])
    assert set(env.movable_dirs[0][21]) == set([1])

    env_run.step(21)
    env_run.step(29)
    env_run.step(22)
    env_run.step(42)
    env_run.step(20)
    env_run.step(14)
    env_run.step(13)
    env_run.step(6)
    env_run.step(7)
    env_run.step(44)
    env_run.step(5)
    env_run.step(38)
    env_run.step(37)
    env_run.step(15)
    env_run.step(23)
    env_run.step(45)
    env_run.step(19)
    env_run.step(12)
    env_run.step(11)
    env_run.step(4)
    env_run.step(3)
    env_run.step(2)
    env_run.render()
    assert env.next_player == 0
    env_run.step(1)
    env_run.render()
    assert env.next_player == 0  # skip
    env_run.step(10)
    env_run.step(18)
    env_run.step(9)
    env_run.step(17)
    env_run.step(0)
    env_run.step(52)
    env_run.step(25)
    env_run.step(8)
    env_run.step(16)
    env_run.step(24)
    env_run.step(32)
    env_run.step(33)
    env_run.step(41)
    env_run.step(48)
    env_run.step(40)
    env_run.step(50)
    env_run.step(56)
    env_run.step(53)
    env_run.step(30)
    env_run.step(46)
    env_run.step(31)
    env_run.step(39)
    env_run.step(47)
    env_run.step(55)
    env_run.step(54)
    env_run.step(63)
    env_run.step(62)
    env_run.step(61)
    env_run.step(60)
    env_run.step(59)
    env_run.step(58)
    env_run.step(57)
    env_run.step(51)
    env_run.step(43)
    env_run.render()
    env_run.step(49)
    env_run.render()
    assert env_run.done
    assert env_run.rewards[0] == 1
    assert env_run.rewards[1] == -1
