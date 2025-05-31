import pytest

from tests.quick.examples.examples_common import setup_examples_test


def test_sample_mlflow():
    pytest.importorskip("mlflow")
    wkdir = setup_examples_test(add_path="")

    import sample_mlflow  # type: ignore

    sample_mlflow.train_ql(timeout=1)
    sample_mlflow.load_ql_parameter()
    sample_mlflow.train_vanilla_policy(timeout=1)
