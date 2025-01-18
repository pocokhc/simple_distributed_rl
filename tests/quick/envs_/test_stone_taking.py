from srl.test.env import env_test, player_test


def test_play():
    env_test("StoneTaking")


def test_player():
    player_test("StoneTaking", "cpu")
