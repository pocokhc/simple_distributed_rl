from srl import runner

# --- env & algorithm load
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


config = runner.Config("Grid", ql.Config())
parameter = config.make_parameter()

rewards = runner.evaluate(config, parameter, max_episodes=5)
print(f"evaluate episodes: {rewards}")
"""
evaluate episodes: [
    -2.0399999544024467,
    -2.079999975860119,
    -1.719999983906746,
    -2.0399999544024467,
    -2.079999975860119
]
"""
