# pip install gym-retro
# gym-retro support python3.6 3.7 3.8 and gym<=0.25.2
import retro

import srl
from srl.algorithms import ql

env_config = srl.EnvConfig(
    "Airstriker-Genesis",
    dict(state="Level1"),
    gym_make_func=retro.make,
)

runner = srl.Runner(env_config, ql.Config())

runner.render_window()
