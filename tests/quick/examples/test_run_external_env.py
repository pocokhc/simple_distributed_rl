# exampleの中身をそのまま使います


from pathlib import Path


def test_run(capfd):
    import os
    import sys

    wkdir = Path(__file__).parent.parent.parent.parent / "examples" / "external_env"
    print(wkdir)
    sys.path.insert(0, str(wkdir))

    os.remove(wkdir / "_parameter.dat")

    # --- 学習
    import srl_train

    srl_train.train()
    assert os.path.isfile(wkdir / "_parameter.dat")

    # --- 評価
    import main

    cap_text = capfd.readouterr().out
    assert cap_text.count("action: [1, 1, 1, 1, 1]") == 5
