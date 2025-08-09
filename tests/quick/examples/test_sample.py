from tests.quick.examples.examples_common import setup_examples_test


def test_sample_basic():
    wkdir = setup_examples_test(add_path="")

    import sample_basic  # type: ignore

    sample_basic.main()
