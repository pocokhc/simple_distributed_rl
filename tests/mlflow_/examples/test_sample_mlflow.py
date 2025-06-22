import os

import pytest

from tests.quick.examples.examples_common import setup_examples_test


def test_sample_mlflow(tmpdir):
    pytest.importorskip("mlflow")
    wkdir = setup_examples_test(add_path="")

    import mlflow
    import sample_mlflow  # type: ignore

    mldir = os.path.join(tmpdir, "mlruns")
    print(mldir)
    mlflow.set_tracking_uri("file:///" + mldir)

    sample_mlflow.train_ql(timeout=1)
    sample_mlflow.load_ql_parameter()
    sample_mlflow.train_vanilla_policy(timeout=1)
