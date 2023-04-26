import os

import pytest  # pip install pytest pytest-cov

if os.path.isfile("coverage.xml"):
    os.remove("coverage.xml")

base_dir = os.path.dirname(__file__)
cov_dir = os.path.join(base_dir, "../srl/")
test_dirs = [
    os.path.join(base_dir, "./other"),
    # os.path.join(base_dir, "./envs_"),
]

os.environ["SRL_MP_SKIP"] = "1"  # mpとpytest-covの相性が悪い？

pytest.main([*test_dirs, f"--cov={cov_dir}", "--cov-report", "xml"])
