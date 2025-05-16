import pytest

from tests.quick.examples.examples_common import setup_examples_test


def test_sample_basic():
    wkdir = setup_examples_test(add_path="")

    import sample_basic  # type: ignore

    sample_basic.main()


def test_sample_commonly():
    pytest.importorskip("gymnasium")
    wkdir = setup_examples_test(add_path="")

    import sample_commonly  # type: ignore

    sample_commonly.train(timeout=1)
    sample_commonly.evaluate()
    sample_commonly.render_terminal()
    # sample_commonly.render_window()
    sample_commonly.animation()
    # sample_commonly.replay_window()


def test_sample_long_training():
    pytest.importorskip("gymnasium")
    wkdir = setup_examples_test(add_path="")

    import sample_long_training  # type: ignore

    sample_long_training.train(timeout=1)
    sample_long_training.evaluate()
