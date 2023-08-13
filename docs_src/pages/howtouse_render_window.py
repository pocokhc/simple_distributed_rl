import srl
from srl.algorithms import ql

runner = srl.Runner("Grid", ql.Config())
runner.train(timeout=5)

runner.render_window()
