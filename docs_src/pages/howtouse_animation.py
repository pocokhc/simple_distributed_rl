import srl
from srl.algorithms import ql

runner = srl.Runner("Grid", ql.Config())

runner.animation_save_gif("_Grid.gif")
# runner.animation_display()  # for notebook
