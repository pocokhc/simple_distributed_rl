# pip install gym-retro
# gym-retro==0.8.0 support python3.6 3.7 3.8 and gym<=0.25.2
import retro

import srl
from srl.utils import common

common.logger_print()

env_config = srl.EnvConfig(
    "Airstriker-Genesis",
    dict(state="Level1"),
    gym_make_func=retro.make,
)

runner = srl.Runner(env_config)
runner.render_window()
