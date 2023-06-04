from srl import runner

# --- env & algorithm load
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


config = runner.Config("Grid", ql.Config())
parameter = config.make_parameter()

runner.render_terminal(config, parameter)
