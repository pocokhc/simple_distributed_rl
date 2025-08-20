import pytest

from tests.quick.examples.examples_common import setup_docs_test


def test_custom_algorithm():
    wkdir = setup_docs_test(add_path="")

    import custom_algorithm1  # type: ignore  # noqa: F401
    import custom_algorithm2  # type: ignore  # noqa: F401
    import custom_algorithm3  # type: ignore  # noqa: F401
    import custom_algorithm4  # type: ignore  # noqa: F401


def test_custom_algorithm4():
    wkdir = setup_docs_test(add_path="")

    import custom_algorithm4  # type: ignore

    custom_algorithm4.main()


def test_custom_env1():
    wkdir = setup_docs_test(add_path="")

    import custom_env1  # type: ignore  # noqa: F401


def test_custom_env2():
    wkdir = setup_docs_test(add_path="")

    import custom_env2  # type: ignore  # noqa: F401


def test_custom_env3():
    wkdir = setup_docs_test(add_path="")

    import custom_env3  # type: ignore  # noqa: F401


def test_custom_env12():
    wkdir = setup_docs_test(add_path="")

    import custom_env12  # type: ignore  # noqa: F401


def test_howtouse():
    wkdir = setup_docs_test(add_path="")

    import howtouse_animation  # type: ignore  # noqa: F401
    import howtouse_eval  # type: ignore  # noqa: F401
    import howtouse_render_terminal  # type: ignore  # noqa: F401
    # import howtouse_render_window  # type: ignore  # noqa: F401


def test_yaml_training():
    wkdir = setup_docs_test(add_path="")

    import yaml_training  # type: ignore  # noqa: F401


def test_yaml_training_hydra():
    pytest.skip("hydraでエラー")
    wkdir = setup_docs_test(add_path="")

    import yaml_training_hydra  # type: ignore  # noqa: F401

    yaml_training_hydra.main()
