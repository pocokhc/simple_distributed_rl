import os

import pytest

from srl.utils.common import is_package_installed
from tests.quick.examples.examples_common import setup_examples_test


def test_env():
    wkdir = setup_examples_test(add_path="raw")

    import env  # type: ignore

    if is_package_installed("pygame"):
        render_mode = "window"
    else:
        render_mode = "terminal"
    env.main("Grid", render_interval=1000 / 180, render_mode=render_mode)
    env.main("OX", render_interval=1000 / 180, render_mode=render_mode)


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
