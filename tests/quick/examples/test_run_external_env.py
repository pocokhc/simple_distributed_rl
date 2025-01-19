import os

from tests.quick.examples.examples_common import setup_examples_test


def test_run(capfd):
    wkdir = setup_examples_test(add_path="external_env")

    # --- 学習
    import srl_train  # type: ignore

    srl_train.train()
    assert os.path.isfile(wkdir / "_parameter.dat")

    # --- 評価
    import main  # type: ignore

    cap_text = capfd.readouterr().out
    assert cap_text.count("action: [1, 1, 1, 1, 1]") == 5
