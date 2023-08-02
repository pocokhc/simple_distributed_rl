import srl
from srl.algorithms import ql

runner = srl.Runner("Grid", ql.Config())

rewards = runner.evaluate(max_episodes=5)
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
