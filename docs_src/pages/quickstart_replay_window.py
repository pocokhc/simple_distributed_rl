from srl import runner

# --- env & algorithm load
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


config = runner.Config("Grid", ql.Config())

parameter, _, _ = runner.train(config, timeout=10)

runner.replay_window(config, parameter)
