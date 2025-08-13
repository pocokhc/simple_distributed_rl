import pytest

from tests.quick.examples.examples_common import setup_examples_test


def test_sample_basic():
    wkdir = setup_examples_test(add_path="")

    import sample_basic  # type: ignore

    sample_basic.main()


def test_sample_commonly():
    wkdir = setup_examples_test(add_path="")

    import sample_commonly  # type: ignore

    sample_commonly.train()
    sample_commonly.evaluate()
    # sample_commonly.render_terminal()
    # sample_commonly.render_window()
    # sample_commonly.animation()


def test_sample_template():
    pytest.skip("長いので一旦保留")
    wkdir = setup_examples_test(add_path="sample_template")

    import main  # type: ignore

    main.train()


def test_sample_template_adv():
    pytest.skip("長いので一旦保留")
    wkdir = setup_examples_test(add_path="sample_template_adv")

    import main  # type: ignore

    main.train()
