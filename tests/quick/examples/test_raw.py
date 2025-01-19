from tests.quick.examples.examples_common import setup_examples_test


def test_env():
    wkdir = setup_examples_test(add_path="raw")

    import env  # type: ignore

    env.main("Grid", render_interval=1000 / 180)
    env.main("OX", render_interval=1000 / 180)


def test_play():
    wkdir = setup_examples_test(add_path="raw")

    import play  # type: ignore

    play.main()


def test_play_2player():
    wkdir = setup_examples_test(add_path="raw")

    import play_2player  # type: ignore

    play_2player.main()
    play_2player.play_cpu("random")


def test_play_use_run():
    wkdir = setup_examples_test(add_path="raw")

    import play_use_run  # type: ignore

    play_use_run.main()


def test_play_mp():
    wkdir = setup_examples_test(add_path="raw")

    import play_mp  # type: ignore

    play_mp.main()


def test_play_mp_use_run():
    wkdir = setup_examples_test(add_path="raw")

    import play_mp_use_run  # type: ignore

    play_mp_use_run.main()
