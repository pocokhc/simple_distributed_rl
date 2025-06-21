import os

import pytest

from tests.quick.examples.examples_common import setup_examples_test

is_github_actions: bool = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(is_github_actions, reason="mpが不安定なため")
def test_play_mp():
    wkdir = setup_examples_test(add_path="raw")

    import play_mp  # type: ignore

    play_mp.main()


@pytest.mark.skipif(is_github_actions, reason="mpが不安定なため")
def test_play_mp_memory():
    wkdir = setup_examples_test(add_path="raw")

    import play_mp_memory  # type: ignore

    play_mp_memory.main()


@pytest.mark.skipif(is_github_actions, reason="mpが不安定なため")
def test_play_mp_no_queue():
    wkdir = setup_examples_test(add_path="raw")

    import play_mp_no_queue  # type: ignore

    play_mp_no_queue.main()
