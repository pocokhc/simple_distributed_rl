import os
import shutil

from srl import runner
from srl.algorithms import ql_agent57
from srl.envs import grid  # noqa F401
from srl.runner.callbacks.checkpoint import Checkpoint
from srl.runner.core import EvalOption, play
from srl.utils import common

common.logger_print()


def test_run():
    dir_name = "tmp_test"
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)

    config = runner.Config("Grid", ql_agent57.Config())
    callback = Checkpoint(
        save_dir=dir_name,
        checkpoint_interval=1,
        eval=EvalOption(),
    )
    play(
        config,
        timeout=3,
        train_only=False,
        enable_profiling=False,
        training=True,
        eval=None,
        history=None,
        checkpoint=None,
        callbacks=[callback],
    )
    assert len(os.listdir(dir_name)) > 0
